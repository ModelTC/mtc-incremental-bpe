use std::borrow::Borrow;

use derive_more::{Constructor, Deref, From, Into};

use crate::{
    NormalizedDict, TokenId,
    aho_corasick::{AC_NODE_ROOT, ACAutomaton, ACNodeId},
    centroid::SufSucCentroidTrees,
    successor::{FOREST_VIRTUAL_ROOT, ForestNodeId, SucForest},
    suf_suc::SufSucNodeSet,
};

#[derive(Debug, Deref)]
pub struct IncBpeTokenizer {
    #[deref]
    dict: NormalizedDict,
    automaton: ACAutomaton,
    forest: SucForest,
    node_set: SufSucNodeSet,
    trees: SufSucCentroidTrees,
}

#[derive(Clone, Copy, Debug, Constructor, Into, From, Hash, PartialEq, Eq, PartialOrd, Ord)]
pub struct IncBpeToken {
    pub token_id: TokenId,
    pub skip_len: u32,
}

impl IncBpeToken {
    pub fn fetch_chain(seq: &[Self], end_pos: usize) -> Vec<usize> {
        if end_pos >= seq.len() {
            return Vec::new();
        }
        let mut positions = vec![end_pos];
        let mut current = end_pos;
        while (seq[current].skip_len as usize) <= current {
            current -= seq[current].skip_len as usize;
            positions.push(current);
        }
        positions
    }

    pub fn token_ids(
        seq: &[Self],
        end_pos: usize,
    ) -> impl IntoIterator<Item = TokenId> + Sync + Send {
        Self::fetch_chain(seq, end_pos)
            .into_iter()
            .rev()
            .map(move |i| seq[i].token_id)
    }
}

impl IncBpeTokenizer {
    pub fn new(dict: NormalizedDict) -> Self {
        let automaton = ACAutomaton::new(&dict);
        let forest = SucForest::new(&dict);
        let node_set = SufSucNodeSet::new(&forest, &automaton);
        let trees = SufSucCentroidTrees::new(&node_set, &forest);
        Self {
            dict,
            automaton,
            forest,
            node_set,
            trees,
        }
    }

    pub fn tokenize<I: IntoIterator<Item = TokenId>>(&self, token_ids: I) -> Vec<IncBpeToken> {
        let mut state = self.tokenization();
        for token_id in token_ids {
            state.feed(token_id);
        }
        state.into_tokens()
    }

    pub fn tokenization(&self) -> IncBpeTokenization<&Self> {
        IncBpeTokenization::new(self)
    }
}

#[derive(Debug)]
pub struct IncBpeTokenization<T> {
    tokenizer: T,
    tokens: Vec<IncBpeToken>,
    forest_ids: Vec<ForestNodeId>,
    ac_state: ACNodeId,
}

impl<T> IncBpeTokenization<T> {
    pub fn new(tokenizer: T) -> Self {
        Self {
            tokenizer,
            tokens: Default::default(),
            forest_ids: Default::default(),
            ac_state: AC_NODE_ROOT,
        }
    }

    pub fn tokens(&self) -> &[IncBpeToken] {
        &self.tokens
    }

    pub fn into_tokens(self) -> Vec<IncBpeToken> {
        self.tokens
    }

    pub fn token_seq(&self, end_pos: usize) -> impl IntoIterator<Item = TokenId> + Sync + Send {
        IncBpeToken::token_ids(&self.tokens, end_pos)
    }

    pub fn current_token_seq(&self) -> impl IntoIterator<Item = TokenId> + Sync + Send {
        self.token_seq(self.tokens.len().saturating_sub(1))
    }
}

impl<T: Borrow<IncBpeTokenizer>> IncBpeTokenization<T> {
    pub fn feed(&mut self, token_id: TokenId) -> IncBpeToken {
        let tokenizer = self.tokenizer.borrow();
        let (token, node_id) = if let Some(token) = tokenizer.get_token(token_id)
            && tokenizer.is_useful(token_id)
        {
            debug_assert!(
                tokenizer.forest[tokenizer.forest.token_to_node_id[token_id]].skip_len == 1
            );
            self.ac_state = tokenizer.automaton.feed(self.ac_state, token);
            let skip_to = |skip| {
                let len = self.forest_ids.len();
                if skip == 0 || skip > len {
                    FOREST_VIRTUAL_ROOT
                } else {
                    self.forest_ids[len - skip]
                }
            };
            let mut forest_id = tokenizer.node_set.longest_token_node[self.ac_state];
            debug_assert!(forest_id != FOREST_VIRTUAL_ROOT);
            let node = &tokenizer.node_set[forest_id];
            if (node.skip_len as usize) <= self.tokens.len() && !node.verify(skip_to) {
                let tree = &tokenizer.trees[forest_id];
                forest_id = tree.search(skip_to);
            }
            let node = &tokenizer.forest[forest_id];
            (IncBpeToken::new(node.token_id, node.skip_len), forest_id)
        } else {
            self.ac_state = AC_NODE_ROOT;
            (IncBpeToken::new(token_id, 1), FOREST_VIRTUAL_ROOT)
        };
        self.tokens.push(token);
        self.forest_ids.push(node_id);
        token
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        Dictionary, IncBpeToken, IncBpeTokenizer, NormalizedDict, TokenId, Vocab,
        sp_impl::sentence_piece_impl,
    };

    fn inc_bpe_short_any_case(vocab: &[&str], rules: &[(&str, &str)], sequences: &[&str]) {
        inc_bpe_short_case::<true>(vocab, rules, sequences);
        inc_bpe_short_case::<false>(vocab, rules, sequences);
    }

    fn inc_bpe_short_case<const IN_BYTES: bool>(
        vocab: &[&str],
        rules: &[(&str, &str)],
        sequences: &[&str],
    ) {
        inc_bpe_case::<IN_BYTES, false>(vocab, rules, sequences);
    }

    fn inc_bpe_display_any_case(vocab: &[&str], rules: &[(&str, &str)], sequences: &[&str]) {
        inc_bpe_display_case::<true>(vocab, rules, sequences);
        inc_bpe_display_case::<false>(vocab, rules, sequences);
    }

    fn inc_bpe_display_case<const IN_BYTES: bool>(
        vocab: &[&str],
        rules: &[(&str, &str)],
        sequences: &[&str],
    ) {
        inc_bpe_case::<IN_BYTES, true>(vocab, rules, sequences);
    }

    fn inc_bpe_case<const IN_BYTES: bool, const DISPLAY: bool>(
        vocab: &[&str],
        rules: &[(&str, &str)],
        sequences: &[&str],
    ) {
        let vocab = Vocab::new(vocab.iter().map(|&s| s.to_owned())).unwrap();
        let dict = Dictionary::new_from_token_pair(vocab, rules.iter().copied()).unwrap();
        let tokenizer = IncBpeTokenizer::new(if IN_BYTES {
            NormalizedDict::new_in_bytes
        } else {
            NormalizedDict::new_in_utf8
        }(dict));

        let validate = |seq: &[_], inc_res: &[IncBpeToken]| {
            for i in 0..seq.len() {
                let expected = sentence_piece_impl::<false>(&tokenizer, seq[0..i + 1].to_vec());
                let output: Vec<_> = IncBpeToken::token_ids(inc_res, i).into_iter().collect();
                assert!(output == expected);
            }
        };

        let tokenize = |s| {
            let single_tokens = if IN_BYTES {
                tokenizer.split_bytes_to_tokens(s, 0usize)
            } else {
                tokenizer.split_utf8_to_tokens(s, 0usize)
            };
            let res = tokenizer.tokenize(single_tokens.iter().copied());
            validate(&single_tokens, &res);
            res
        };

        let display_res = |res: &[IncBpeToken]| {
            if DISPLAY {
                let msg = String::from_iter(res.iter().map(|t| {
                    let token = str::from_utf8(&tokenizer[t.token_id]).unwrap();
                    format!("[{token} ({})], ", t.token_id)
                }));
                println!("{msg}");
            }
        };

        for s in sequences {
            let res = tokenize(s);
            display_res(&res);
        }
    }

    #[test]
    fn test_inc_bpe_short() {
        let vocab = [
            "<unk>", "a", "abc", "abcde", "abcdef", "b", "ba", "bc", "bcdef", "c", "cd", "cde",
            "cdefg", "d", "de", "def", "e", "ef", "efg", "f", "g",
        ];
        inc_bpe_display_any_case(
            &vocab,
            &[
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
            &["abcdefg", "babcdefg", "cdefg"],
        );
        inc_bpe_display_any_case(
            &vocab,
            &[
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
            &["abcdefg", "babcdefg", "cdefg"],
        );

        let vocab = ["<unk>", "a", "aa", "aaa", "aaaa", "aaaaa"];
        let rules = [("a", "a"), ("aa", "a"), ("aa", "aa"), ("aa", "aaa")];
        let seq = [
            "a", "aa", "aaa", "aaaa", "aaaaa", "aaaaaa", "aaaaaaa", "aaaaaaaa",
        ];
        inc_bpe_short_any_case(&vocab, &rules, &seq);
        let rules = [("a", "a"), ("aa", "aa"), ("aa", "a"), ("aaaa", "a")];
        inc_bpe_short_any_case(&vocab, &rules, &seq);
        let rules = [("a", "a")];
        inc_bpe_display_any_case(&vocab, &rules, &seq);
        let rules = [("a", "a"), ("a", "aa")];
        inc_bpe_short_any_case(&vocab, &rules, &seq);

        for i in 1..6 {
            let mut vocab = vec!["<unk>".to_owned()];
            vocab.extend((0..i).map(|i| String::from_iter(std::iter::repeat_n("a", i + 1))));
            let vocab: Vec<_> = vocab.iter().map(|s| s.as_str()).collect();
            let all_rules: Vec<_> = vocab
                .iter()
                .skip(1)
                .flat_map(|s| (1..s.len()).map(|p| s.split_at(p)))
                .collect();
            assert!(all_rules.len() <= (1 << 10));
            eprintln!("{vocab:?} {all_rules:?}");
            for j in 0..(1 << all_rules.len()) {
                let rules: Vec<_> = all_rules
                    .iter()
                    .enumerate()
                    .filter_map(|(k, &v)| if (j & (1 << k)) != 0 { Some(v) } else { None })
                    .collect();
                inc_bpe_short_any_case(&vocab, &rules, &seq);
            }
        }

        let vocab = ["<unk>", "a", "aa", "aaa", "aaaa", "aaaaa"];
        let rules = [("a", "a"), ("aa", "a"), ("aa", "aa"), ("aa", "aaa")];
        let mut multiple_a_s: Vec<_> = [
            "a", "aa", "aaa", "aaaa", "aaaaa", "aaaaaa", "aaaaaaa", "aaaaaaaa",
        ]
        .map(|s| s.to_owned())
        .into_iter()
        .collect();
        for _ in 0..3 {
            for s in multiple_a_s.clone() {
                for k in 0..s.len() + 1 {
                    let (u, v) = s.split_at(k);
                    multiple_a_s.push(String::from_iter([u, "b", v]));
                }
            }
        }
        eprintln!("{multiple_a_s:?}");
        let multiple_a_s = multiple_a_s.iter().map(|s| s.as_str()).collect::<Vec<_>>();
        inc_bpe_short_any_case(&vocab, &rules, &multiple_a_s);
        let rules = [("a", "a"), ("aa", "aa"), ("aa", "a"), ("aaaa", "a")];
        inc_bpe_short_any_case(&vocab, &rules, &multiple_a_s);
        let rules = [("a", "a")];
        inc_bpe_short_any_case(&vocab, &rules, &multiple_a_s);
        let rules = [("a", "a"), ("a", "aa")];
        inc_bpe_short_any_case(&vocab, &rules, &multiple_a_s);

        let vocab = [
            "<unk>",
            "a",
            "b",
            "c",
            "d",
            "cd",
            "bcd",
            "abcd",
            "你",
            "好",
            "呀",
            "你好",
            "你好呀",
            "好你",
            "aa",
            "aaa",
        ];
        inc_bpe_short_any_case(
            &vocab,
            &[("c", "d"), ("b", "cd"), ("a", "bcd")],
            &["dcdbcdabcdab"],
        );
        inc_bpe_short_case::<false>(
            &vocab,
            &[("你", "好")],
            &["你好", "你好呀", "你好你好你好呀你好你好你"],
        );
        inc_bpe_short_case::<false>(
            &vocab,
            &[("你", "好"), ("你好", "呀")],
            &["你好", "你好呀", "你好你好你好呀你好你好你", "", "你"],
        );
        let seq = ["好你好你好呀你好你好你", "你好你好你好呀你好你好你"];
        for rules in [
            [("你", "好"), ("你好", "呀"), ("好", "你")],
            [("你", "好"), ("好", "你"), ("你好", "呀")],
            [("好", "你"), ("你", "好"), ("你好", "呀")],
        ] {
            inc_bpe_short_case::<false>(&vocab, &rules, &seq);
        }

        for rules in [
            &[("a", "a")] as &[_],
            &[("a", "a"), ("aa", "a")],
            &[("a", "a"), ("a", "aa")],
            &[("aa", "a"), ("a", "a")],
        ] {
            inc_bpe_short_any_case(&vocab, rules, &multiple_a_s);
        }
    }

    fn inc_bpe_demo_case(rules: &[(&str, &str)]) {
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

        let dict = Dictionary::new_from_token_pair(vocab, rules.iter().copied()).unwrap();
        let tokenizer = IncBpeTokenizer::new(NormalizedDict::new_in_bytes(dict));
        let tokenize = |s| tokenizer.tokenize(tokenizer.split_bytes_to_tokens(s, 0usize));

        let display_res = |res: &[IncBpeToken]| {
            let msg = String::from_iter(res.iter().map(|t| {
                let token = str::from_utf8(&tokenizer[t.token_id]).unwrap();
                format!("[{token} ({})], ", t.token_id)
            }));
            println!("{msg}");
        };

        println!("{rules:?}");
        let res = tokenize("abcdefg");
        display_res(&res);
        let res = tokenize("babcdefg");
        display_res(&res);
        let res = tokenize("cdefg");
        display_res(&res);
    }

    #[test]
    fn test_inc_bpe_non_vocab_token() {
        let vocab = Vocab::new(["a", "aa"]).unwrap();
        let avail_token_ids = [0, 2, 3, TokenId::MAX.inner()].map(TokenId::new);
        for rules in [&[] as &[_], &[(0usize, 0usize)]] {
            let dict = Dictionary::new_from_id_pair(vocab.clone(), rules.iter().copied()).unwrap();
            let tokenizer = IncBpeTokenizer::new(NormalizedDict::new_in_bytes(dict));
            let validate = |seq: &[_], inc_res: &[IncBpeToken]| {
                for i in 0..seq.len() {
                    let expected = sentence_piece_impl::<false>(&tokenizer, seq[0..i + 1].to_vec());
                    let output = {
                        IncBpeToken::token_ids(inc_res, i)
                            .into_iter()
                            .collect::<Vec<_>>()
                    };
                    assert!(output == expected);
                }
            };
            for len in 1..9 {
                for seq in 0..(1 << (len * 2)) {
                    let token_ids = (0..len)
                        .map(|i| avail_token_ids[(seq >> (i * 2)) & 3])
                        .collect::<Vec<_>>();
                    let res = tokenizer.tokenize(token_ids.iter().copied());
                    validate(&token_ids, &res);
                }
            }
        }
    }

    #[test]
    fn test_inc_bpe_demo() {
        inc_bpe_demo_case(&[
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
        ]);
        inc_bpe_demo_case(&[
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
        ]);
    }
}
