use std::{borrow::Borrow, collections::VecDeque};

use derive_more::Debug;

use crate::{
    IncBpeToken, IncBpeTokenization, IncBpeTokenizer, SkipLen, TokenId,
    aho_corasick::{AC_NODE_ROOT, ACNodeId},
    successor::{FOREST_VIRTUAL_ROOT, ForestNodeId},
};

#[derive(Debug)]
struct EagerTokenNode {
    forest_id: ForestNodeId,
    token_id: TokenId,
    skip_len: SkipLen,
    num_alive_children: u16,
    feed_len: u16,
}

#[derive(Debug)]
pub struct EagerBpeTokenization<T> {
    #[debug(ignore)]
    tokenizer: T,
    nodes: VecDeque<EagerTokenNode>,
    useful_offset: u16,
    num_useful_bytes: u16,
    num_roots: u16,
    ac_state: ACNodeId,
}

impl IncBpeTokenizer {
    pub fn eager(&self) -> EagerBpeTokenization<&Self> {
        EagerBpeTokenization {
            tokenizer: self,
            nodes: Default::default(),
            useful_offset: 0,
            num_useful_bytes: 0,
            num_roots: 0,
            ac_state: AC_NODE_ROOT,
        }
    }
}

impl<T> From<EagerBpeTokenization<T>> for IncBpeTokenization<T> {
    fn from(value: EagerBpeTokenization<T>) -> Self {
        let capacity = value.nodes.len();
        let mut forest_ids = Vec::with_capacity(capacity);
        let mut tokens = Vec::with_capacity(capacity);
        for node in value.nodes {
            forest_ids.push(node.forest_id);
            tokens.push(IncBpeToken::const_new(node.token_id, node.skip_len));
        }
        Self::new_internal(value.tokenizer, value.ac_state, tokens, forest_ids)
    }
}

impl<T> EagerBpeTokenization<T> {
    fn pop_prefix_removed_nodes(&mut self) {
        while self.useful_offset > 0
            && self
                .nodes
                .front()
                .is_some_and(|i| i.num_alive_children == 0)
        {
            self.nodes.pop_front();
            self.useful_offset -= 1;
        }
    }

    fn move_forward_useful_offset(&mut self) {
        debug_assert!(self.useful_offset as usize + 1 < self.nodes.len());
        let mut idx = self.useful_offset;
        self.useful_offset += 1;
        self.num_useful_bytes -= self.nodes[idx as usize].feed_len;
        loop {
            let node = &self.nodes[idx as usize];
            if node.num_alive_children != 0 || idx < node.skip_len {
                if node.num_alive_children == 0 {
                    debug_assert!(self.num_roots > 1);
                    self.num_roots -= 1;
                }
                break;
            }
            idx -= node.skip_len;
            self.nodes[idx as usize].num_alive_children -= 1;
        }
    }
}

impl<T: Borrow<IncBpeTokenizer>> EagerBpeTokenization<T> {
    fn maintain_useful_offset(&mut self) {
        let tokenizer: &IncBpeTokenizer = self.tokenizer.borrow();
        let target_useful_bytes = tokenizer.ac_depths[self.ac_state];
        while self.useful_offset as usize + 1 < self.nodes.len()
            && self.num_useful_bytes
                > target_useful_bytes + self.nodes[self.useful_offset as usize].feed_len
        {
            self.move_forward_useful_offset();
        }
    }

    fn push(&mut self, forest_id: ForestNodeId, feed_len: u16) {
        let tokenizer: &IncBpeTokenizer = self.tokenizer.borrow();
        let suc_node = &tokenizer.forest[forest_id];
        let token_id = suc_node.token_id;
        let skip_len = suc_node.skip_len;
        if self.nodes.len() < skip_len as usize {
            self.num_roots += 1;
        } else {
            let parent = self.nodes.len() - skip_len as usize;
            self.nodes[parent].num_alive_children += 1;
        }
        self.num_useful_bytes += feed_len;
        self.nodes.push_back(EagerTokenNode {
            forest_id,
            token_id,
            feed_len,
            skip_len,
            num_alive_children: 0,
        });
    }
}

impl<T: Borrow<IncBpeTokenizer>> EagerBpeTokenization<T> {
    pub fn feed(&mut self, token_id: TokenId) {
        let tokenizer: &IncBpeTokenizer = self.tokenizer.borrow();
        if let Some(token) = tokenizer.get_token(token_id)
            && tokenizer.is_useful(token_id)
        {
            #[cfg(debug_assertions)]
            {
                let node_id = tokenizer.forest.token_to_node_id[token_id];
                debug_assert_eq!(tokenizer.forest[node_id].skip_len, 1);
            }
            self.ac_state = tokenizer.trans_table.feed(self.ac_state, token);
            let feed_len = token.len() as u16;
            let skip_to = |skip| {
                let len = self.nodes.len();
                if skip == 0 || skip > len {
                    FOREST_VIRTUAL_ROOT
                } else {
                    self.nodes[len - skip].forest_id
                }
            };
            let mut forest_id = tokenizer.node_set.longest_token_node[self.ac_state];
            debug_assert_ne!(forest_id, FOREST_VIRTUAL_ROOT);
            let node = &tokenizer.node_set[forest_id];
            if (node.skip_len as usize) <= self.nodes.len() && !node.verify(skip_to) {
                let tree = &tokenizer.trees[forest_id];
                forest_id = tree.search(skip_to);
            }
            self.push(forest_id, feed_len);
            self.maintain_useful_offset();
            self.pop_prefix_removed_nodes();
        } else {
            self.ac_state = AC_NODE_ROOT;
            while self.useful_offset as usize + 1 < self.nodes.len() {
                self.move_forward_useful_offset();
            }
            self.pop_prefix_removed_nodes();
            if let Some(node) = self.nodes.back_mut() {
                debug_assert_eq!(node.num_alive_children, 0);
                debug_assert_eq!(self.num_roots, 1);
                node.num_alive_children = 1;
            } else {
                debug_assert_eq!(self.num_roots, 0);
                self.num_roots = 1;
            }
            self.useful_offset = self.nodes.len() as _;
            self.num_useful_bytes = 0;
            self.nodes.push_back(EagerTokenNode {
                forest_id: FOREST_VIRTUAL_ROOT,
                token_id,
                skip_len: 1,
                num_alive_children: 0,
                feed_len: 0,
            });
        }
    }
}

impl<T> EagerBpeTokenization<T> {
    pub fn new(tokenizer: T) -> Self {
        Self {
            tokenizer,
            nodes: Default::default(),
            useful_offset: 0,
            num_useful_bytes: 0,
            num_roots: 0,
            ac_state: AC_NODE_ROOT,
        }
    }

    pub fn reset(&mut self) {
        self.nodes.clear();
        self.useful_offset = 0;
        self.num_useful_bytes = 0;
        self.num_roots = 0;
        self.ac_state = AC_NODE_ROOT;
    }

    pub fn reserve(&mut self, additional: usize) {
        self.nodes.reserve(additional);
    }

    pub fn tokenizer(&self) -> &T {
        &self.tokenizer
    }
}

impl<T> Iterator for EagerBpeTokenization<T> {
    type Item = IncBpeToken;

    fn next(&mut self) -> Option<Self::Item> {
        if self.num_roots != 1 {
            return None;
        }
        self.pop_prefix_removed_nodes();
        if self.useful_offset == 0 {
            return None;
        }
        let EagerTokenNode {
            forest_id: _,
            feed_len: _,
            token_id,
            skip_len,
            num_alive_children,
        } = self.nodes.pop_front()?;
        self.useful_offset -= 1;
        self.num_roots = num_alive_children;
        Some(IncBpeToken::const_new(token_id, skip_len))
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        Dictionary, IncBpeToken, IncBpeTokenizer, NormalizedDict, TokenId, Vocab,
        test_utils::{bpe_with_heap, bytes_into_tokens, utf8_into_tokens},
    };

    fn eager_bpe_any_case(vocab: &[&str], rules: &[(&str, &str)], sequences: &[&str]) {
        eager_bpe_short_case::<true>(vocab, rules, sequences);
        eager_bpe_short_case::<false>(vocab, rules, sequences);
    }

    fn eager_bpe_short_case<const IN_BYTES: bool>(
        vocab: &[&str],
        rules: &[(&str, &str)],
        sequences: &[&str],
    ) {
        eager_bpe_case::<IN_BYTES, false>(vocab, rules, sequences);
    }

    fn eager_bpe_display_any_case(vocab: &[&str], rules: &[(&str, &str)], sequences: &[&str]) {
        eager_bpe_display_case::<true>(vocab, rules, sequences);
        eager_bpe_display_case::<false>(vocab, rules, sequences);
    }

    fn eager_bpe_display_case<const IN_BYTES: bool>(
        vocab: &[&str],
        rules: &[(&str, &str)],
        sequences: &[&str],
    ) {
        eager_bpe_case::<IN_BYTES, true>(vocab, rules, sequences);
    }

    fn validate(dict: &Dictionary, seq: &[TokenId], eager_res: &[IncBpeToken]) {
        let expected = bpe_with_heap::<false>(dict, seq);
        let output: Vec<_> = eager_res.iter().map(|&t| t.token_id).collect();
        assert_eq!(output, expected);
    }

    fn eager_bpe_case<const IN_BYTES: bool, const DISPLAY: bool>(
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

            let mut state = tokenizer.eager();
            let mut output = Vec::new();
            for token_id in std::iter::chain(single_tokens.iter().copied(), [TokenId::MAX]) {
                state.feed(token_id);
                output.extend(&mut state);
            }

            let mut batch_state = tokenizer.eager();
            let mut batch_output = Vec::new();
            for token_ids in std::iter::chain(single_tokens.chunks(4), [TokenId::MAX].chunks(1)) {
                for token_id in token_ids.iter().copied() {
                    batch_state.feed(token_id);
                }
                batch_output.extend(&mut batch_state);
            }
            assert_eq!(output, batch_output);

            validate(&tokenizer, &single_tokens, &output);
            output
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
    fn test_eager_bpe_unk_tokens() {
        eager_bpe_display_any_case(
            &["", "a", "b", "ab", "ba", "aa"],
            &[("a", "b"), ("b", "a"), ("a", "a")],
            &["acbacbcabbacaaaaaacccabaccabca", "ccc", "c", ""],
        );
    }

    #[test]
    fn test_eager_bpe_short() {
        let vocab = [
            "", "a", "abc", "abcde", "abcdef", "b", "ba", "bc", "bcdef", "c", "cd", "cde", "cdefg",
            "d", "de", "def", "e", "ef", "efg", "f", "g",
        ];
        eager_bpe_display_any_case(
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
        eager_bpe_display_any_case(
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
        eager_bpe_any_case(&vocab, &rules, &seq);
        let rules = [("a", "a"), ("aa", "aa"), ("aa", "a"), ("aaaa", "a")];
        eager_bpe_any_case(&vocab, &rules, &seq);
        let rules = [("a", "a")];
        eager_bpe_display_any_case(&vocab, &rules, &seq);
        let rules = [("a", "a"), ("a", "aa")];
        eager_bpe_any_case(&vocab, &rules, &seq);

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
                eager_bpe_any_case(&vocab, &rules, &seq);
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
        eager_bpe_any_case(&vocab, &rules, &multiple_a_s);
        let rules = [("a", "a"), ("aa", "aa"), ("aa", "a"), ("aaaa", "a")];
        eager_bpe_any_case(&vocab, &rules, &multiple_a_s);
        let rules = [("a", "a")];
        eager_bpe_any_case(&vocab, &rules, &multiple_a_s);
        let rules = [("a", "a"), ("a", "aa")];
        eager_bpe_any_case(&vocab, &rules, &multiple_a_s);

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
        eager_bpe_any_case(
            &vocab,
            &[("c", "d"), ("b", "cd"), ("a", "bcd")],
            &["dcdbcdabcdab"],
        );
        eager_bpe_short_case::<false>(
            &vocab,
            &[("你", "好")],
            &["你好", "你好呀", "你好你好你好呀你好你好你"],
        );
        eager_bpe_short_case::<false>(
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
            eager_bpe_short_case::<false>(&vocab, &rules, &seq);
        }

        for rules in [
            &[("a", "a")] as &[_],
            &[("a", "a"), ("aa", "a")],
            &[("a", "a"), ("a", "aa")],
            &[("aa", "a"), ("a", "a")],
        ] {
            eager_bpe_any_case(&vocab, rules, &multiple_a_s);
        }
    }

    #[test]
    fn test_eager_bpe_non_longest() {
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
        eager_bpe_display_any_case(&vocab, &rules, &sequences);
    }

    fn eager_bpe_demo_case(rules: &[(&str, &str)]) {
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
            let init_token_seq = bytes_into_tokens(&tokenizer, s, 0usize);
            let mut tokenization = tokenizer.eager();
            let mut res = Vec::new();
            for token_id in std::iter::chain(init_token_seq, [TokenId::MAX]) {
                tokenization.feed(token_id);
                res.extend(&mut tokenization);
            }
            res
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
    fn test_eager_bpe_non_vocab_token() {
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
                    let mut tokenization = tokenizer.eager();
                    let mut res = Vec::new();
                    for token_id in std::iter::chain(token_ids.iter().copied(), [TokenId::MAX]) {
                        tokenization.feed(token_id);
                        res.extend(&mut tokenization);
                    }
                    validate(&tokenizer, &token_ids, &res);
                }
            }
        }
    }

    #[test]
    fn test_eager_bpe_demo() {
        eager_bpe_demo_case(&[
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
        eager_bpe_demo_case(&[
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

    #[test]
    fn test_eager_bpe_repeated() {
        let vocab: Vec<String> = ["".to_owned()]
            .into_iter()
            .chain((1..=32).map(|i| std::iter::repeat_n('a', i).collect()))
            .collect();
        let vocab_ref: Vec<_> = vocab.iter().map(|s| s.as_ref()).collect();
        eager_bpe_display_any_case(
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
