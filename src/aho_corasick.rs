use std::{
    collections::VecDeque,
    ops::{Index, IndexMut},
};

use derive_more::{Debug, Deref};
use tinyvec::TinyVec;

use crate::{
    TokenId, Vocab,
    typed_vec::{TypedVec, typed_vec_index},
};

typed_vec_index!(pub(crate) ACNodeId, u32);

pub(crate) const AC_NODE_ROOT: ACNodeId = ACNodeId::ZERO;

const ALPHABET_SIZE: usize = 1 << 8;

pub(crate) type ACNodeIdVec = TinyVec<[ACNodeId; 8]>;
type ByteKeyVec = TinyVec<[u8; 14]>;

const _: () = {
    assert!(std::mem::size_of::<ACNodeIdVec>() == 40);
    assert!(std::mem::size_of::<ByteKeyVec>() == 24);
};

#[derive(Debug)]
struct ACNode {
    #[debug(ignore)]
    map: [ACNodeId; ALPHABET_SIZE],
    children: ACNodeIdVec,
    keys: ByteKeyVec,
}

const _: () = {
    assert!(std::mem::size_of::<ACNode>() == 64 + 4 * ALPHABET_SIZE);
    assert!(std::mem::size_of::<[ACNode; 2]>() == std::mem::size_of::<ACNode>() * 2);
};

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

    #[inline(always)]
    fn index(&self, index: u8) -> &Self::Output {
        &self.map[index as usize]
    }
}

impl IndexMut<u8> for ACNode {
    #[inline(always)]
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
    #[inline(always)]
    pub fn num_of_nodes(&self) -> ACNodeId {
        self.nodes.len()
    }

    fn get_or_add(&mut self, node_id: ACNodeId, byte: u8) -> ACNodeId {
        let child = self.nodes[node_id][byte];
        if child == AC_NODE_ROOT {
            let child_id = self.nodes.push(Default::default());
            self.nodes[node_id].add_child(byte, child_id);
            child_id
        } else {
            child
        }
    }

    #[inline(always)]
    fn get(&self, node_id: ACNodeId, byte: u8) -> ACNodeId {
        self.nodes[node_id][byte]
    }

    #[inline(always)]
    fn children_nodes(&self, node_id: ACNodeId) -> &[ACNodeId] {
        &self.nodes[node_id].children
    }

    pub fn children(&self, node_id: ACNodeId) -> impl Iterator<Item = (ACNodeId, u8)> {
        let node = &self.nodes[node_id];
        debug_assert_eq!(node.children.len(), node.keys.len());
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

        let mut token_to_node = TypedVec::with_capacity(vocab.num_of_tokens().as_usize());

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
            let mut suffix_children = TypedVec::from(vec![ACNodeIdVec::new(); len]);
            for (cur, suf) in suffix.enumerate_copied() {
                if cur == suf {
                    debug_assert_eq!(cur, AC_NODE_ROOT);
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
            let (cur_node, suf_node) = trie.nodes.two_diff_mut(id, suf);
            for (cur, suf) in cur_node.map.iter_mut().zip(&suf_node.map) {
                if *cur == AC_NODE_ROOT {
                    *cur = *suf;
                }
            }
        }

        Self {
            trie,
            suffix,
            token_to_node,
        }
    }

    #[inline(always)]
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
            println!("{node:2} {suffix:2}: {children:?}");
        }
        for (id, token) in vocab.tokens.enumerate() {
            let node = automaton.token_to_node[id];
            let suffix = automaton.suffix[node];
            println!("{node:2} {suffix:2}: {}", str::from_utf8(token).unwrap());
        }
    }
}
