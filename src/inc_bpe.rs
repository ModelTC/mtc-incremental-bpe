use std::borrow::Borrow;

use derive_more::{Constructor, Deref, From, Into};

use crate::{
    NormalizedDict, SkipLen, TokenId,
    aho_corasick::{AC_NODE_ROOT, ACAutomaton, ACNodeId, ACTransTable},
    centroid::SufSucCentroidTrees,
    successor::{FOREST_VIRTUAL_ROOT, ForestNodeId, SucForest},
    suf_suc::SufSucNodeSet,
};

#[derive(Clone, Copy, Debug, Into, From, Hash, PartialEq, Eq, PartialOrd, Ord)]
pub struct IncBpeToken {
    pub token_id: TokenId,
    pub skip_len: SkipLen,
}

#[derive(Debug, Deref)]
pub struct IncBpeTokenizer {
    #[deref]
    dict: NormalizedDict,
    automaton: ACTransTable,
    forest: SucForest,
    node_set: SufSucNodeSet,
    trees: SufSucCentroidTrees,
}

#[derive(Clone, Debug, Constructor, Into, From)]
pub struct IncBpeTokenChainIter<S> {
    seq: S,
    pos: usize,
}

#[derive(Debug)]
pub struct IncBpeTokenization<T> {
    tokenizer: T,
    tokens: Vec<IncBpeToken>,
    forest_ids: Vec<ForestNodeId>,
    ac_state: ACNodeId,
}

impl IncBpeToken {
    pub fn new<I: Into<TokenId>>(token_id: I, skip_len: SkipLen) -> Self {
        Self::new_const(token_id.into(), skip_len)
    }

    pub fn new_const(token_id: TokenId, skip_len: SkipLen) -> Self {
        Self { token_id, skip_len }
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
            automaton: automaton.trans_table,
            forest,
            node_set,
            trees,
        }
    }

    pub fn tokenize<I: IntoIterator<Item = TokenId>>(
        &self,
        token_ids: I,
    ) -> IncBpeTokenization<&Self> {
        let iter = token_ids.into_iter();
        let mut state = self.tokenization();
        state.reserve(iter.size_hint().0);
        for token_id in iter {
            state.feed(token_id);
        }
        state
    }

    pub fn tokenization(&self) -> IncBpeTokenization<&Self> {
        IncBpeTokenization::new(self)
    }
}

impl<S> IncBpeTokenChainIter<S> {
    pub fn pos(&self) -> usize {
        self.pos
    }

    pub fn seq(&self) -> &S {
        &self.seq
    }
}

impl<S: Borrow<[IncBpeToken]>> IncBpeTokenChainIter<S> {
    pub fn token_ids(self) -> impl Iterator<Item = TokenId> {
        self.map(|(_, t)| t.token_id)
    }
}

impl<S: Borrow<[IncBpeToken]>> Iterator for IncBpeTokenChainIter<S> {
    type Item = (usize, IncBpeToken);

    fn next(&mut self) -> Option<Self::Item> {
        let seq: &[IncBpeToken] = self.seq.borrow();
        let pos = self.pos;
        if pos >= seq.len() {
            return None;
        }
        let token = seq[pos];
        let skip_len = token.skip_len as usize;
        if skip_len <= pos {
            self.pos -= skip_len;
        } else {
            debug_assert_eq!(skip_len, pos + 1);
            self.pos = seq.len();
        }
        Some((pos, token))
    }
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
}

impl<T: Borrow<IncBpeTokenizer>> IncBpeTokenization<T> {
    pub fn feed(&mut self, token_id: TokenId) -> IncBpeToken {
        let tokenizer: &IncBpeTokenizer = self.tokenizer.borrow();
        let (token, node_id) = if let Some(token) = tokenizer.get_token(token_id)
            && tokenizer.is_useful(token_id)
        {
            #[cfg(debug_assertions)]
            {
                let node_id = tokenizer.forest.token_to_node_id[token_id];
                debug_assert_eq!(tokenizer.forest[node_id].skip_len, 1);
            }
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
            debug_assert_ne!(forest_id, FOREST_VIRTUAL_ROOT);
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

impl<T> IncBpeTokenization<T> {
    pub fn reset(&mut self) {
        self.tokens.clear();
        self.forest_ids.clear();
        self.ac_state = AC_NODE_ROOT;
    }

    pub fn reserve(&mut self, additional: usize) {
        self.tokens.reserve(additional);
        self.forest_ids.reserve(additional);
    }

    pub fn inc_tokens(&self) -> &[IncBpeToken] {
        &self.tokens
    }

    pub fn tokenizer(&self) -> &T {
        &self.tokenizer
    }

    pub fn into_inner(self) -> (T, Vec<IncBpeToken>) {
        (self.tokenizer, self.tokens)
    }

    pub fn into_inc_tokens(self) -> Vec<IncBpeToken> {
        self.tokens
    }

    pub fn token_chain(&self, end_pos: usize) -> IncBpeTokenChainIter<&[IncBpeToken]> {
        IncBpeTokenChainIter::new(&self.tokens, end_pos)
    }

    pub fn current_token_chain(&self) -> IncBpeTokenChainIter<&[IncBpeToken]> {
        self.token_chain(self.tokens.len().saturating_sub(1))
    }
}

#[cfg(test)]
mod tests {
    use std::{borrow::Borrow, sync::Arc};

    use crate::{
        Dictionary, IncBpeToken, IncBpeTokenChainIter, IncBpeTokenizer, NormalizedDict, TokenId,
        Vocab,
        test_utils::{bpe_with_heap, bytes_into_tokens, utf8_into_tokens},
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

    fn validate(dict: &Dictionary, seq: &[TokenId], inc_res: &[IncBpeToken]) {
        for i in 0..seq.len() {
            let expected = bpe_with_heap::<false>(dict, &seq[0..i + 1]);
            let output = IncBpeTokenChainIter::new(inc_res, i).token_ids();
            let output = output.chain(std::iter::repeat(TokenId::MAX));
            assert!(expected.into_iter().rev().zip(output).all(|(i, j)| i == j));
        }
    }

    fn inc_bpe_case<const IN_BYTES: bool, const DISPLAY: bool>(
        vocab: &[&str],
        rules: &[(&str, &str)],
        sequences: &[&str],
    ) {
        let vocab = Vocab::new(vocab.iter().map(|&s| s.to_owned())).unwrap();
        let dict = Dictionary::new_from_token_pair(vocab, rules.iter().copied()).unwrap();
        let tokenizer = IncBpeTokenizer::new(
            if IN_BYTES {
                NormalizedDict::new_in_bytes
            } else {
                NormalizedDict::new_in_utf8
            }(dict)
            .unwrap(),
        );

        let tokenize = |s: &str| {
            let single_tokens = if IN_BYTES {
                bytes_into_tokens(&tokenizer, s, 0usize)
            } else {
                utf8_into_tokens(&tokenizer, s, 0usize)
            };
            let res = tokenizer
                .tokenize(single_tokens.iter().copied())
                .into_inc_tokens();
            validate(&tokenizer, &single_tokens, &res);
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
            "", "a", "abc", "abcde", "abcdef", "b", "ba", "bc", "bcdef", "c", "cd", "cde", "cdefg",
            "d", "de", "def", "e", "ef", "efg", "f", "g",
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

        let vocab = ["", "a", "aa", "aaa", "aaaa", "aaaaa"];
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
            for j in 0..(1 << all_rules.len()) {
                let rules: Vec<_> = all_rules
                    .iter()
                    .enumerate()
                    .filter_map(|(k, &v)| if (j & (1 << k)) != 0 { Some(v) } else { None })
                    .collect();
                inc_bpe_short_any_case(&vocab, &rules, &seq);
            }
        }

        let vocab = ["", "a", "aa", "aaa", "aaaa", "aaaaa"];
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
        let multiple_a_s: Vec<_> = multiple_a_s.iter().map(|s| s.as_str()).collect();
        inc_bpe_short_any_case(&vocab, &rules, &multiple_a_s);
        let rules = [("a", "a"), ("aa", "aa"), ("aa", "a"), ("aaaa", "a")];
        inc_bpe_short_any_case(&vocab, &rules, &multiple_a_s);
        let rules = [("a", "a")];
        inc_bpe_short_any_case(&vocab, &rules, &multiple_a_s);
        let rules = [("a", "a"), ("a", "aa")];
        inc_bpe_short_any_case(&vocab, &rules, &multiple_a_s);

        let vocab = [
            "",
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

    #[test]
    fn test_inc_bpe_non_longest() {
        let vocab = [
            "", "a", "b", "c", "d", "e", "f", "g", "h", "i", "ab", "ba", "bc", "cd", "de", "ef",
            "gh", "hi", "cde", "ghi", "fghi", "abcd", "fg", "efgh", "efghi", "bcd", "defgh",
            "bcde", "bcdef", "bcdefgh",
        ];
        let rules = [
            ("b", "a"),
            ("a", "b"),
            ("e", "f"),
            ("f", "g"),
            ("d", "e"),
            ("c", "de"),
            ("c", "d"),
            ("b", "cde"),
            ("b", "c"),
            ("b", "cd"),
            ("ab", "cd"),
            ("g", "h"),
            ("h", "i"),
            ("gh", "i"),
            ("ef", "gh"),
            ("d", "efgh"),
            ("bcd", "ef"),
            ("bcd", "efgh"),
            ("fg", "hi"),
            ("ef", "ghi"),
        ];
        let mut sequences = vec!["babcdefghi"];
        while sequences.last().unwrap().len() > 1 {
            sequences.push(&sequences.last().unwrap()[1..])
        }
        {
            let vocab = Vocab::new(vocab.iter().map(|&s| s.to_owned())).unwrap();
            let dict =
                Dictionary::new_from_token_pair(vocab.clone(), rules.iter().copied()).unwrap();
            let normalized = NormalizedDict::new_in_bytes(dict).unwrap();
            let mut expected: Vec<_> = normalized
                .useful_rules
                .values()
                .map(|i| i.as_usize())
                .collect();
            expected.sort();
            assert_eq!(expected, (0..rules.len()).collect::<Vec<_>>());
            assert!(
                vocab
                    .tokens
                    .keys()
                    .skip(1)
                    .all(|id| normalized.is_useful(id))
            );
        }
        inc_bpe_display_any_case(&vocab, &rules, &sequences);
    }

    fn inc_bpe_demo_case(rules: &[(&str, &str)]) {
        let vocab = Vocab::new([
            b"" as &[_],
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
        let tokenizer = IncBpeTokenizer::new(NormalizedDict::new_in_bytes(dict).unwrap());
        let tokenize = |s| {
            tokenizer
                .tokenize(bytes_into_tokens(&tokenizer, s, 0usize))
                .into_inc_tokens()
        };

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
            let tokenizer = IncBpeTokenizer::new(NormalizedDict::new_in_bytes(dict).unwrap());
            for len in 1..9 {
                for seq in 0..(1 << (len * 2)) {
                    let token_ids: Vec<_> = (0..len)
                        .map(|i| avail_token_ids[(seq >> (i * 2)) & 3])
                        .collect();
                    let res = tokenizer
                        .tokenize(token_ids.iter().copied())
                        .into_inc_tokens();
                    validate(&tokenizer, &token_ids, &res);
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

    fn verify_chain_iter(
        chain: impl Borrow<[IncBpeToken]> + Clone,
        seqs: impl IntoIterator<Item = impl IntoIterator<Item = impl Into<TokenId>>>,
    ) {
        for (end_pos, seq) in seqs.into_iter().enumerate() {
            let expected: Vec<_> = seq.into_iter().map(|i| i.into()).collect();
            let iter = IncBpeTokenChainIter::new(chain.clone(), end_pos);
            let output: Vec<_> = iter.map(|(_, i)| i.token_id).collect();
            assert_eq!(output, expected);
        }
    }

    #[test]
    fn test_inc_bpe_chain_iter_box() {
        let chain_base = [
            IncBpeToken::new(1u32, 1),
            IncBpeToken::new(2u32, 1),
            IncBpeToken::new(3u32, 2),
            IncBpeToken::new(4u32, 2),
            IncBpeToken::new(5u32, 1),
            IncBpeToken::new(6u32, 3),
        ];

        let expected = [
            &[1u32] as &[u32],
            &[2, 1],
            &[3, 1],
            &[4, 2, 1],
            &[5, 4, 2, 1],
            &[6, 3, 1],
        ];
        let build_seq = || expected.iter().map(|&s| s.iter().copied());

        verify_chain_iter(&chain_base as &[IncBpeToken], build_seq());
        verify_chain_iter(chain_base, build_seq());

        let chain: Vec<IncBpeToken> = Vec::from(chain_base);
        verify_chain_iter(chain, build_seq());

        let chain: Box<[IncBpeToken]> = Box::from(chain_base);
        verify_chain_iter(chain, build_seq());

        let chain: Arc<[IncBpeToken]> = Arc::from(chain_base);
        verify_chain_iter(chain, build_seq());
    }

    #[test]
    fn test_inc_bpe_repeated() {
        let vocab: Vec<String> = ["".to_owned()]
            .into_iter()
            .chain((1..=32).map(|i| std::iter::repeat_n('a', i).collect()))
            .collect();
        let vocab_ref: Vec<_> = vocab.iter().map(|s| s.as_ref()).collect();
        inc_bpe_display_any_case(
            &vocab_ref[..18],
            &[
                ("a", "a"),
                ("aa", "a"),
                ("aa", "aa"),
                ("aaaa", "aaaa"),
                ("aaaa", "aa"),
                ("aa", "aaa"),
                ("aaaa", "aaa"),
                ("aaaaaaaa", "aaaaaaaa"),
            ],
            &vocab_ref[1..],
        );
    }
}
