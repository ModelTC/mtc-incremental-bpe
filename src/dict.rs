use std::{hash::Hash, ops::Index};

use bytes::BytesMut;
use derive_more::{Deref, Display, From, Into};
use rapidhash::{HashMapExt, RapidHashMap};
use thiserror::Error;
use tinyvec::TinyVec;

use crate::{
    Token, TokenId, Vocab,
    typed_vec::{TypedVec, typed_vec_index},
};

typed_vec_index!(pub RuleId, u32);

type TokenRules = TinyVec<[RuleId; 6]>;

const _: () = {
    assert!(std::mem::size_of::<TokenRules>() == 32);
};

#[derive(Clone, Debug, Hash, PartialEq, Eq, PartialOrd, Ord, Into, From)]
pub struct Rule {
    pub merged: TokenId,
    pub pre: TokenId,
    pub suc: TokenId,
}

#[derive(Clone, Debug, Deref)]
pub struct Dictionary {
    #[deref]
    vocab: Vocab,
    pub(crate) rules: TypedVec<RuleId, Rule>,
    token_to_rule_ids: TypedVec<TokenId, TokenRules>,
    pair_to_rule_id: RapidHashMap<(TokenId, TokenId), RuleId>,
}

#[derive(Clone, Debug, Display)]
#[display("{:?}", self.0.as_ref())]
pub struct UnknownToken(pub Token);

#[derive(Clone, Debug, Error)]
#[non_exhaustive]
pub enum DictBuildError {
    #[error("the {pos}-th rule contains an unknown token {token}")]
    UnknownToken { pos: RuleId, token: UnknownToken },
    #[error("the {pos}-th rule contains an unknown token id {id}")]
    UnknownTokenId { pos: RuleId, id: TokenId },
    #[error("the {pos}-th rule contains an empty token id {id}")]
    EmptyToken { pos: RuleId, id: TokenId },
}

impl Dictionary {
    fn from_rules(vocab: Vocab, rules: TypedVec<RuleId, Rule>) -> Self {
        let mut token_to_rule_ids = TypedVec::from_iter(std::iter::repeat_n(
            TokenRules::new(),
            vocab.num_of_tokens().as_usize(),
        ));
        let mut pair_to_rule_id = RapidHashMap::with_capacity(rules.len().as_usize());
        for (id, rule) in rules.enumerate() {
            token_to_rule_ids[rule.merged].push(id);
            pair_to_rule_id.insert((rule.pre, rule.suc), id);
        }
        Self {
            vocab,
            rules,
            token_to_rule_ids,
            pair_to_rule_id,
        }
    }

    pub fn new_from_id_pair<T: Into<TokenId>, R: IntoIterator<Item = (T, T)>>(
        vocab: Vocab,
        rule_iter: R,
    ) -> Result<Self, DictBuildError> {
        let rule_iter = rule_iter.into_iter();
        let mut rules = TypedVec::with_capacity(RuleId::from(rule_iter.size_hint().0));
        let get_token = |pos, id| {
            vocab
                .get_token(id)
                .ok_or(DictBuildError::UnknownTokenId { pos, id })
                .and_then(|t| {
                    if t.is_empty() {
                        Err(DictBuildError::EmptyToken { pos, id })
                    } else {
                        Ok(t)
                    }
                })
        };
        for (pos, (left, right)) in rule_iter.map(|(i, j)| (i.into(), j.into())).enumerate() {
            let pos = RuleId::from(pos);
            let token = {
                let mut buf = BytesMut::from(get_token(pos, left)?.clone());
                buf.extend_from_slice(get_token(pos, right)?);
                buf.freeze()
            };
            let merged = vocab
                .find_token_id(&token)
                .ok_or(DictBuildError::UnknownToken {
                    pos,
                    token: UnknownToken(token),
                })?;
            rules.push(Rule {
                merged,
                pre: left,
                suc: right,
            });
        }
        Ok(Self::from_rules(vocab, rules))
    }

    pub fn new_from_token_pair<T: AsRef<[u8]>, R: IntoIterator<Item = (T, T)>>(
        vocab: Vocab,
        rule_iter: R,
    ) -> Result<Self, DictBuildError> {
        let rule_iter = rule_iter.into_iter();
        let mut rules = TypedVec::with_capacity(RuleId::from(rule_iter.size_hint().0));
        let get_id = |pos, token: &[u8]| {
            vocab
                .find_token_id(token)
                .ok_or(DictBuildError::UnknownToken {
                    pos,
                    token: UnknownToken(token.to_owned().into()),
                })
        };
        for (pos, (left, right)) in rule_iter.enumerate() {
            let (left, right) = (left.as_ref(), right.as_ref());
            let pos = RuleId::from(pos);
            let left_id = get_id(pos, left)?;
            let right_id = get_id(pos, right)?;
            let token = {
                let mut buf = BytesMut::from(left);
                buf.extend_from_slice(right);
                buf.freeze()
            };
            let merged = get_id(pos, &token)?;
            rules.push(Rule {
                merged,
                pre: left_id,
                suc: right_id,
            });
        }
        Ok(Self::from_rules(vocab, rules))
    }

    pub fn rules(&self) -> &[Rule] {
        self.rules.as_slice()
    }

    pub fn get_rule(&self, rule_id: RuleId) -> Option<&Rule> {
        self.rules.get(rule_id)
    }

    pub fn num_of_rules(&self) -> RuleId {
        self.rules.len()
    }

    pub fn find_rule(&self, left: TokenId, right: TokenId) -> Option<RuleId> {
        self.pair_to_rule_id.get(&(left, right)).copied()
    }

    pub fn get_rule_ids(&self, token_id: TokenId) -> Option<&[RuleId]> {
        self.token_to_rule_ids.get(token_id).map(|v| v.as_slice())
    }

    pub fn is_proper<F: FnMut(TokenId, &[u8]) -> bool>(
        &self,
        mut is_single: F,
    ) -> Result<(), RuleId> {
        let mut check = move |rule_id: RuleId, sub_token_id: TokenId| -> bool {
            self.token_to_rule_ids[sub_token_id]
                .first()
                .is_some_and(|&sub_rule_id| sub_rule_id < rule_id)
                || is_single(sub_token_id, &self[sub_token_id])
        };
        for (rule_id, rule) in self.rules.enumerate() {
            if !check(rule_id, rule.pre) || !check(rule_id, rule.suc) {
                return Err(rule_id);
            }
        }
        Ok(())
    }

    pub fn is_proper_in_bytes(&self) -> Result<(), RuleId> {
        self.is_proper(|_, b| b.len() == 1)
    }

    pub fn is_proper_in_utf8(&self) -> Result<(), RuleId> {
        self.is_proper(|_, b| {
            if b.len() > 4 {
                return false;
            }
            std::str::from_utf8(b).is_ok_and(|s| s.chars().count() == 1)
        })
    }
}

impl Index<RuleId> for Dictionary {
    type Output = Rule;

    #[inline(always)]
    fn index(&self, index: RuleId) -> &Self::Output {
        self.rules.index(index)
    }
}

impl Index<TokenId> for Dictionary {
    type Output = Token;

    #[inline(always)]
    fn index(&self, index: TokenId) -> &Self::Output {
        self.vocab.index(index)
    }
}

#[cfg(test)]
mod tests {
    use crate::{Dictionary, Vocab};

    fn build_dict<T: AsRef<[u8]>, R: IntoIterator<Item = (T, T)>>(
        vocab: &Vocab,
        rules: R,
    ) -> Dictionary {
        Dictionary::new_from_token_pair(vocab.clone(), rules).unwrap()
    }

    #[test]
    fn test_dict() {
        let vocab = Vocab::new([
            b"a" as &[_],
            b"b",
            b"c",
            b"d",
            b"cd",
            b"bcd",
            b"abcd",
            "你".as_bytes(),
            "好".as_bytes(),
            "呀".as_bytes(),
            "你好".as_bytes(),
            "你好呀".as_bytes(),
            b"\xe4",
            b"\xbd",
            b"\xa0",
            b"\xbd\xa0",
        ])
        .unwrap();

        assert!(Dictionary::new_from_token_pair(vocab.clone(), [("c", "d")]).is_ok());
        assert!(Dictionary::new_from_token_pair(vocab.clone(), [("a", "b")]).is_err());
        assert!(Dictionary::new_from_id_pair(vocab.clone(), [(2usize, 3)]).is_ok());
        assert!(Dictionary::new_from_id_pair(vocab.clone(), [(0usize, 1)]).is_err());

        let dict = build_dict(&vocab, [("c", "d"), ("b", "cd"), ("a", "bcd")]);
        assert!(dict.is_proper_in_bytes().is_ok());
        assert!(dict.is_proper_in_utf8().is_ok());

        let dict = build_dict(&vocab, [("b", "cd")]);
        assert!(dict.is_proper_in_bytes().is_err());
        assert!(dict.is_proper_in_utf8().is_err());

        let dict = build_dict(
            &vocab,
            [(b"\xbd" as &[_], b"\xa0" as &[_]), (b"\xe4", b"\xbd\xa0")],
        );
        assert!(dict.is_proper_in_bytes().is_ok());
        assert!(dict.is_proper_in_utf8().is_err());

        let dict = build_dict(&vocab, [("你", "好")]);
        assert!(dict.is_proper_in_bytes().is_err());
        assert!(dict.is_proper_in_utf8().is_ok());

        let dict = build_dict(&vocab, [("你", "好"), ("你好", "呀")]);
        assert!(dict.is_proper_in_bytes().is_err());
        assert!(dict.is_proper_in_utf8().is_ok());

        let dict = build_dict(&vocab, [("你好", "呀"), ("你", "好")]);
        assert!(dict.is_proper_in_bytes().is_err());
        assert!(dict.is_proper_in_utf8().is_err());
    }
}
