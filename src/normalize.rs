use std::collections::BTreeMap;

use derive_more::Deref;

use crate::{Dictionary, RuleId, TokenId, typed_vec::TypedVec};

pub(crate) const SINGLETON_PRIORITY: RuleId = {
    let mut priority = RuleId::MAX;
    *priority.inner_mut() = (priority.inner() >> 1) + 1;
    priority
};

fn singleton_token_id(rule_id: RuleId) -> TokenId {
    debug_assert!(rule_id >= SINGLETON_PRIORITY);
    TokenId::new((rule_id - SINGLETON_PRIORITY).inner())
}

#[derive(Clone, Debug, Deref)]
pub struct NormalizedDict {
    #[deref]
    dict: Dictionary,
    pub(crate) priorities: TypedVec<TokenId, RuleId>,
    #[cfg(test)]
    pub(crate) useful_rules: BTreeMap<(TokenId, TokenId), RuleId>,
}

impl NormalizedDict {
    pub fn new<F: FnMut(&Dictionary, TokenId, &[u8]) -> bool>(
        dict: Dictionary,
        mut is_single: F,
    ) -> Self {
        let len = dict.num_of_tokens().as_usize();
        let mut priorities = TypedVec::<TokenId, _>::from(vec![RuleId::MAX; len]);
        let mut useful_rules: BTreeMap<(TokenId, TokenId), RuleId> = Default::default();

        for (id, priority) in priorities.enumerate_mut() {
            if is_single(&dict, id, &dict[id]) {
                debug_assert!(id.as_usize() < SINGLETON_PRIORITY.as_usize());
                let mut p = SINGLETON_PRIORITY;
                *p.inner_mut() += id.inner();
                *priority = p;
            }
        }

        'outer: for (id, rule) in dict.rules.enumerate() {
            let mut left = priorities[rule.pre];
            let mut right = priorities[rule.suc];
            if priorities[rule.merged] != RuleId::MAX || left == RuleId::MAX || right == RuleId::MAX
            {
                continue;
            }
            while left < SINGLETON_PRIORITY || right < SINGLETON_PRIORITY {
                let (u, v): (TokenId, TokenId);
                if left == right {
                    u = dict[left].suc;
                    v = dict[right].pre;
                } else if left >= SINGLETON_PRIORITY {
                    u = singleton_token_id(left);
                    v = dict[right].pre;
                    debug_assert_eq!(left, priorities[u]);
                } else if right >= SINGLETON_PRIORITY {
                    u = dict[left].suc;
                    v = singleton_token_id(right);
                    debug_assert_eq!(right, priorities[v]);
                } else if left > right {
                    u = dict[left].suc;
                    v = dict[right].merged;
                    debug_assert_eq!(right, priorities[v]);
                } else {
                    u = dict[left].merged;
                    v = dict[right].pre;
                    debug_assert_eq!(left, priorities[u]);
                }
                if let Some(&mid) = useful_rules.get(&(u, v)) {
                    debug_assert!(priorities[u] >= SINGLETON_PRIORITY || mid > priorities[u]);
                    debug_assert!(priorities[v] >= SINGLETON_PRIORITY || mid > priorities[v]);
                    if left == right || right == priorities[v] {
                        if mid < left {
                            continue 'outer;
                        }
                    } else if mid <= right {
                        continue 'outer;
                    }
                }
                if left < SINGLETON_PRIORITY {
                    left = priorities[u];
                }
                if right < SINGLETON_PRIORITY {
                    right = priorities[v];
                }
                debug_assert_ne!(left, RuleId::MAX);
                debug_assert_ne!(right, RuleId::MAX);
            }
            priorities[rule.merged] = id;
            let res = useful_rules.insert((rule.pre, rule.suc), id);
            debug_assert!(res.is_none());
        }

        Self {
            dict,
            priorities,
            #[cfg(test)]
            useful_rules,
        }
    }

    pub fn new_in_bytes(dict: Dictionary) -> Self {
        Self::new(dict, |_, _, b| b.len() == 1)
    }

    pub fn new_in_utf8(dict: Dictionary) -> Self {
        Self::new(dict, |_, _, b| {
            if b.len() > 4 {
                return false;
            }
            std::str::from_utf8(b).is_ok_and(|s| s.chars().count() == 1)
        })
    }

    pub fn is_single(&self, id: TokenId) -> bool {
        self.priorities[id] != RuleId::MAX && self.priorities[id] >= SINGLETON_PRIORITY
    }

    pub fn is_useful(&self, id: TokenId) -> bool {
        self.priorities[id] != RuleId::MAX
    }
}

#[cfg(test)]
mod tests {
    use crate::{Dictionary, NormalizedDict, RuleId, Vocab, sp_impl::sentence_piece_impl};

    fn build_dict<T: AsRef<[u8]>, R: IntoIterator<Item = (T, T)>>(
        vocab: &Vocab,
        rules: R,
    ) -> Dictionary {
        Dictionary::new_from_token_pair(vocab.clone(), rules).unwrap()
    }

    fn build_in_bytes(dict: &Dictionary) -> NormalizedDict {
        let dict = NormalizedDict::new_in_bytes(dict.clone());
        for rule in &dict.rules {
            let token_id = rule.merged;
            assert!(!dict.is_single(token_id));
            let seq = &dict[token_id];
            let res = sentence_piece_impl::<false>(&dict, dict.split_bytes_to_tokens(seq, 0usize));
            assert!(dict.is_useful(token_id) ^ (res != vec![token_id]));
        }
        dict
    }

    fn build_in_utf8(dict: &Dictionary) -> NormalizedDict {
        let dict = NormalizedDict::new_in_utf8(dict.clone());
        for rule in &dict.rules {
            let token_id = rule.merged;
            let seq = match std::str::from_utf8(&dict[token_id]) {
                Ok(seq) => seq,
                Err(_) => {
                    assert!(!dict.is_useful(token_id));
                    continue;
                }
            };
            assert!(!dict.is_single(token_id));
            let res = sentence_piece_impl::<false>(&dict, dict.split_utf8_to_tokens(seq, 0usize));
            assert!(dict.is_useful(token_id) ^ (res != vec![token_id]));
        }
        dict
    }

    fn useful_rules<R: IntoIterator<Item = u32>>(dict: &NormalizedDict, rules: R) {
        let mut rules: Vec<_> = rules.into_iter().map(RuleId::new).collect();
        rules.sort();
        let mut expected: Vec<_> = dict.useful_rules.values().copied().collect();
        expected.sort();
        assert_eq!(rules, expected);
    }

    fn build_and_test_rules<R: IntoIterator<Item = u32> + Clone>(
        dict: &Dictionary,
        rules: R,
    ) -> NormalizedDict {
        let normalized = build_in_bytes(dict);
        useful_rules(&normalized, rules.clone());
        let normalized = build_in_utf8(dict);
        useful_rules(&normalized, rules);
        normalized
    }

    #[test]
    fn test_normalized_dict() {
        let vocab = Vocab::new([
            b"<unk>" as &[_],
            b"a",
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
            "好你".as_bytes(),
            b"\xe4",
            b"\xbd",
            b"\xa0",
            b"\xbd\xa0",
            b"aa",
            b"aaa",
            b"aaaa",
            b"aaaaa",
        ])
        .unwrap();

        let dict = build_dict(&vocab, [("c", "d"), ("b", "cd"), ("a", "bcd")]);
        build_and_test_rules(&dict, [0, 1, 2]);

        let dict = build_dict(
            &vocab,
            [(b"\xbd" as &[_], b"\xa0" as &[_]), (b"\xe4", b"\xbd\xa0")],
        );
        let normalized = build_in_bytes(&dict);
        useful_rules(&normalized, [0, 1]);

        let dict = build_dict(&vocab, [("aa", "a"), ("a", "a")]);
        build_and_test_rules(&dict, [1]);

        let dict = build_dict(&vocab, [("a", "aa"), ("a", "a")]);
        build_and_test_rules(&dict, [1]);

        let dict = build_dict(&vocab, [("a", "a"), ("aa", "a")]);
        build_and_test_rules(&dict, [0, 1]);

        let dict = build_dict(&vocab, [("a", "a"), ("a", "aa")]);
        build_and_test_rules(&dict, [0]);

        let dict = build_dict(
            &vocab,
            [
                ("a", "a"),
                ("aa", "a"),
                ("a", "aa"),
                ("aa", "aa"),
                ("a", "aaa"),
                ("aaa", "a"),
            ],
        );
        build_and_test_rules(&dict, [0, 1, 3]);

        let dict = build_dict(&vocab, [("a", "a"), ("aa", "a"), ("aaa", "a")]);
        build_and_test_rules(&dict, [0, 1]);

        let dict = build_dict(&vocab, [("a", "a"), ("aa", "a"), ("aa", "aa")]);
        build_and_test_rules(&dict, [0, 1, 2]);
        let dict = build_dict(&vocab, [("a", "a"), ("aa", "aa"), ("aa", "a")]);
        build_and_test_rules(&dict, [0, 1, 2]);

        let dict = build_dict(
            &vocab,
            [
                ("a", "a"),
                ("aa", "aa"),
                ("aa", "a"),
                ("aaa", "aa"),
                ("aa", "aaa"),
                ("aaaa", "a"),
            ],
        );
        build_and_test_rules(&dict, [0, 1, 2, 5]);

        let dict = build_dict(
            &vocab,
            [
                ("a", "a"),
                ("aa", "a"),
                ("aa", "aa"),
                ("aaa", "aa"),
                ("aa", "aaa"),
                ("aaaa", "a"),
            ],
        );
        build_and_test_rules(&dict, [0, 1, 2, 4]);

        let dict = build_dict(&vocab, [("你", "好"), ("你好", "呀")]);
        let normalized = build_in_utf8(&dict);
        useful_rules(&normalized, [0, 1]);
        let dict = build_dict(&vocab, [("你", "好"), ("你好", "呀"), ("好", "你")]);
        let normalized = build_in_utf8(&dict);
        useful_rules(&normalized, [0, 1, 2]);
        let dict = build_dict(&vocab, [("你", "好"), ("好", "你"), ("你好", "呀")]);
        let normalized = build_in_utf8(&dict);
        useful_rules(&normalized, [0, 1, 2]);
        let dict = build_dict(&vocab, [("好", "你"), ("你", "好"), ("你好", "呀")]);
        let normalized = build_in_utf8(&dict);
        useful_rules(&normalized, [0, 1, 2]);
        let dict = build_dict(&vocab, [("你好", "呀"), ("你", "好"), ("好", "你")]);
        let normalized = build_in_utf8(&dict);
        useful_rules(&normalized, [1, 2]);
        let dict = build_dict(&vocab, [("你好", "呀"), ("好", "你"), ("你", "好")]);
        let normalized = build_in_utf8(&dict);
        useful_rules(&normalized, [1, 2]);
        let dict = build_dict(&vocab, [("好", "你"), ("你好", "呀"), ("你", "好")]);
        let normalized = build_in_utf8(&dict);
        useful_rules(&normalized, [0, 2]);

        let vocab = Vocab::new([
            b"<unk>" as &[_],
            b"a",
            b"abc",
            b"abcde",
            b"abcdef",
            b"b",
            b"ba",
            b"bc",
            b"bcdef",
            b"c",
            b"cd",
            b"cde",
            b"cdefg",
            b"d",
            b"de",
            b"def",
            b"e",
            b"ef",
            b"efg",
            b"f",
            b"g",
        ])
        .unwrap();
        let dict = build_dict(
            &vocab,
            [
                ("b", "c"),
                ("e", "f"),
                ("d", "e"),
                ("c", "d"),
                ("d", "ef"),
                ("b", "a"),
                ("a", "bc"),
                ("abc", "de"),
                ("abc", "def"),
                ("bc", "def"),
                ("c", "de"),
                ("ef", "g"),
                ("cd", "efg"),
            ],
        );
        build_and_test_rules(&dict, 0..13);
        let dict = build_dict(
            &vocab,
            [
                ("b", "c"),
                ("e", "f"),
                ("d", "e"),
                ("c", "d"),
                ("d", "ef"),
                ("a", "bc"),
                ("b", "a"),
                ("abc", "de"),
                ("abc", "def"),
                ("bc", "def"),
                ("c", "de"),
                ("ef", "g"),
                ("cd", "efg"),
            ],
        );
        build_and_test_rules(&dict, 0..13);
    }
}
