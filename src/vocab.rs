use std::{collections::HashMap, hash::Hash, ops::Index};

use bytes::Bytes;
use thiserror::Error;

use crate::typed_vec::{TypedVec, typed_vec_index};

typed_vec_index!(pub TokenId, u32);

pub type Token = Bytes;

#[derive(Clone, Debug)]
pub struct Vocab {
    pub(crate) tokens: TypedVec<TokenId, Token>,
    token_to_id: HashMap<Token, TokenId>,
}

#[derive(Clone, Debug, Error)]
pub enum VocabBuildError {
    #[error("the token with id {id} is empty")]
    EmptyToken { id: TokenId },
    #[error("duplicated tokens with id {a} and {b}")]
    Duplicated { a: TokenId, b: TokenId },
}

impl Vocab {
    pub fn new<T: Into<Token>, I: IntoIterator<Item = T>>(
        iter: I,
    ) -> Result<Self, VocabBuildError> {
        let mut token_to_id = HashMap::new();
        let tokens: TypedVec<_, _> = iter
            .into_iter()
            .enumerate()
            .map(|(k, token)| {
                let token = token.into();
                let id = TokenId::from(k);
                if token.is_empty() {
                    Err(VocabBuildError::EmptyToken { id })
                } else if let Some(other) = token_to_id.insert(token.clone(), id) {
                    Err(VocabBuildError::Duplicated { a: other, b: id })
                } else {
                    Ok(token)
                }
            })
            .collect::<Result<_, _>>()?;
        debug_assert!(tokens.as_slice().len() == token_to_id.len());
        Ok(Self {
            tokens,
            token_to_id,
        })
    }

    pub fn get_token_id<T: AsRef<[u8]>>(&self, token: T) -> Option<TokenId> {
        self.token_to_id.get(token.as_ref()).copied()
    }

    pub fn get_token(&self, id: TokenId) -> Option<&Token> {
        self.tokens.get(id)
    }

    pub fn num_of_tokens(&self) -> TokenId {
        self.tokens.len()
    }

    pub fn tokens(&self) -> &[Token] {
        self.tokens.as_slice()
    }

    pub fn token_to_id_map(&self) -> &HashMap<Token, TokenId> {
        &self.token_to_id
    }

    pub fn split_bytes_to_tokens<S: AsRef<[u8]>>(
        &self,
        seq: S,
        unknown_token_id: impl Into<TokenId>,
    ) -> Vec<TokenId> {
        let unknown_token_id = unknown_token_id.into();
        seq.as_ref()
            .windows(1)
            .map(|i| self.get_token_id(i).unwrap_or(unknown_token_id))
            .collect()
    }

    pub fn split_utf8_to_tokens<S: AsRef<str>>(
        &self,
        seq: S,
        unknown_token_id: impl Into<TokenId>,
    ) -> Vec<TokenId> {
        let unknown_token_id = unknown_token_id.into();
        let seq = seq.as_ref();
        let mut left = 0;
        let mut res = Vec::with_capacity(seq.len());
        for right in 1..seq.len() + 1 {
            if !seq.is_char_boundary(right) {
                continue;
            }
            res.push(
                self.get_token_id(&seq.as_bytes()[left..right])
                    .unwrap_or(unknown_token_id),
            );
            left = right;
        }
        debug_assert!(res.len() == seq.chars().count());
        res
    }
}

impl Index<TokenId> for Vocab {
    type Output = Token;

    fn index(&self, index: TokenId) -> &Self::Output {
        self.tokens.index(index)
    }
}

#[cfg(test)]
mod tests {
    use crate::{TokenId, Vocab};

    #[test]
    fn test_vocab() {
        assert!(Vocab::new([b"abc" as &[_], b"abcd"]).is_ok());
        assert!(Vocab::new([b"a" as &[_], b"", b"b"]).is_err());

        let vocab = Vocab::new([b"a" as &[_], b"b", b"c", b"d", b"cd", b"bcd", b"abcd"]).unwrap();

        assert_eq!(vocab.num_of_tokens().0, 7);

        assert_eq!(vocab.get_token_id(b"a"), Some(TokenId::new(0)));
        assert_eq!(vocab.get_token_id(b"b"), Some(TokenId::new(1)));
        assert_eq!(vocab.get_token_id(b"cd"), Some(TokenId::new(4)));
        assert_eq!(vocab.get_token_id(b"abcd"), Some(TokenId::new(6)));
        assert_eq!(vocab.get_token_id(b""), None);
        assert_eq!(vocab.get_token_id(b"e"), None);
        assert_eq!(vocab.get_token_id(b"random"), None);

        assert!(
            vocab
                .get_token(0_u32.into())
                .is_some_and(|t| t.as_ref() == b"a")
        );
        assert!(
            vocab
                .get_token(3_u32.into())
                .is_some_and(|t| t.as_ref() == b"d")
        );
        assert!(
            vocab
                .get_token(6_u32.into())
                .is_some_and(|t| t.as_ref() == b"abcd")
        );
        assert!(vocab.get_token(7_u32.into()).is_none());
    }

    #[test]
    fn test_pre_tokenize() {
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

        assert_eq!(
            vocab.split_bytes_to_tokens("你好", vocab.num_of_tokens()),
            [
                12,
                13,
                14,
                vocab.num_of_tokens().inner(),
                vocab.num_of_tokens().inner(),
                13,
            ]
            .map(TokenId::new),
        );

        assert_eq!(
            vocab.split_utf8_to_tokens("你好", vocab.num_of_tokens()),
            [7, 8].map(TokenId::new),
        );
    }
}
