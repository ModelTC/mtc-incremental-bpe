use std::{
    collections::VecDeque,
    ops::{Index, IndexMut},
};

use derive_more::{Debug, Deref};
use smallvec::SmallVec;

use crate::{
    TokenId, Vocab,
    typed_vec::{TypedVec, typed_vec_index},
};

typed_vec_index!(pub(crate) ACNodeId, u32);

pub(crate) const AC_NODE_ROOT: ACNodeId = ACNodeId::ZERO;

pub(crate) const NUM_INLINE_AC_NODES: usize = 8;

const ALPHABET_SIZE: usize = 1 << 8;

#[derive(Debug)]
struct ACNode {
    #[debug(ignore)]
    map: [ACNodeId; ALPHABET_SIZE],
    children: SmallVec<[ACNodeId; NUM_INLINE_AC_NODES]>,
    keys: SmallVec<[u8; NUM_INLINE_AC_NODES]>,
}

impl Default for ACNode {
    fn default() -> Self {
        Self {
            map: [AC_NODE_ROOT; _],
            children: Default::default(),
            keys: Default::default(),
        }
    }
}

impl ACNode {
    fn add_child(&mut self, byte: u8, child: ACNodeId) {
        self[byte] = child;
        self.children.push(child);
        self.keys.push(byte);
    }
}

impl Index<u8> for ACNode {
    type Output = ACNodeId;

    fn index(&self, index: u8) -> &Self::Output {
        &self.map[index as usize]
    }
}

impl IndexMut<u8> for ACNode {
    fn index_mut(&mut self, index: u8) -> &mut Self::Output {
        &mut self.map[index as usize]
    }
}

#[derive(Debug)]
pub(crate) struct ACTrie {
    nodes: TypedVec<ACNodeId, ACNode>,
}

impl Default for ACTrie {
    fn default() -> Self {
        Self {
            nodes: vec![Default::default()].into(),
        }
    }
}

impl ACTrie {
    pub fn num_of_nodes(&self) -> ACNodeId {
        self.nodes.len()
    }

    fn get_or_add(&mut self, id: ACNodeId, byte: u8) -> ACNodeId {
        let child = self.nodes[id][byte];
        if child == AC_NODE_ROOT {
            let child_id = self.nodes.push(Default::default());
            self.nodes[id].add_child(byte, child_id);
            child_id
        } else {
            child
        }
    }

    fn get(&self, id: ACNodeId, byte: u8) -> ACNodeId {
        self.nodes[id][byte]
    }

    fn children_nodes(&self, id: ACNodeId) -> &[ACNodeId] {
        &self.nodes[id].children
    }

    pub fn children(&self, id: ACNodeId) -> impl Iterator<Item = (ACNodeId, u8)> {
        let node = &self.nodes[id];
        node.children.iter().copied().zip(node.keys.iter().copied())
    }
}

#[derive(Debug, Deref)]
pub(crate) struct ACAutomaton {
    #[deref]
    trie: ACTrie,
    pub suffix: TypedVec<ACNodeId, ACNodeId>,
    pub token_to_node: TypedVec<TokenId, ACNodeId>,
}

impl ACAutomaton {
    pub fn new(vocab: &Vocab) -> Self {
        let mut trie = ACTrie::default();

        let mut token_to_node = TypedVec::default();

        for token in vocab.tokens.iter() {
            let mut node = AC_NODE_ROOT;
            for &byte in token.as_ref() {
                node = trie.get_or_add(node, byte);
            }
            token_to_node.push(node);
        }

        let len = trie.num_of_nodes().as_usize();
        let mut suffix = TypedVec::from(vec![AC_NODE_ROOT; len]);
        let mut queue = VecDeque::with_capacity(len);
        queue.extend(trie.children_nodes(AC_NODE_ROOT).iter().copied());

        while let Some(node) = queue.pop_front() {
            for (child, byte) in trie.children(node) {
                let mut cursor = suffix[node];
                while cursor != AC_NODE_ROOT && trie.get(cursor, byte) == AC_NODE_ROOT {
                    cursor = suffix[cursor];
                }
                suffix[child] = trie.get(cursor, byte);
                queue.push_back(child);
            }
        }

        let order = {
            let mut suffix_children = TypedVec::from(vec![
                    SmallVec::<[ACNodeId; NUM_INLINE_AC_NODES]>::new();
                    len
                ]);
            for (cur, suf) in suffix.enumerate_copied() {
                if cur == suf {
                    debug_assert!(cur == AC_NODE_ROOT);
                    continue;
                }
                suffix_children[suf].push(cur);
            }
            let mut order = vec![AC_NODE_ROOT];
            let mut current = 0;
            while current < order.len() {
                order.extend(suffix_children[order[current]].iter().copied());
                current += 1;
            }
            order
        };

        for id in order {
            let suf = suffix[id];
            if suf == id {
                continue;
            }
            for b in 0..ALPHABET_SIZE {
                if trie.nodes[id][b as u8] == AC_NODE_ROOT {
                    trie.nodes[id][b as u8] = trie.nodes[suf][b as u8];
                }
            }
        }

        Self {
            trie,
            suffix,
            token_to_node,
        }
    }

    pub fn feed<B: AsRef<[u8]>>(&self, mut node: ACNodeId, bytes: B) -> ACNodeId {
        for &byte in bytes.as_ref() {
            node = self.get(node, byte);
        }
        node
    }
}

#[cfg(test)]
mod tests {
    use crate::{Vocab, aho_corasick::ACAutomaton};

    #[test]
    fn test_ac_automaton() {
        let vocab = Vocab::new([
            b"a" as &[u8],
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
        let automaton = ACAutomaton::new(&vocab);
        for node in automaton.nodes.keys() {
            let suffix = automaton.suffix[node];
            let children: Vec<_> = automaton.children(node).collect();
            println!("{node:2}: {suffix:2}, {children:?}");
        }
        for (id, token) in vocab.tokens.enumerate() {
            let node = automaton.token_to_node[id];
            let suffix = automaton.suffix[node];
            println!("{node:2} {suffix:2}: {}", str::from_utf8(token).unwrap());
        }
    }
}
