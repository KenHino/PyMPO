use pyo3::prelude::*;
use std::collections::{HashMap, HashSet};


#[pyfunction]
fn get_min_vertex_cover(
    u: HashSet<String>,
    e: HashSet<(String, String)>,
    max_matching: HashMap<String, String>,
) -> Vec<String> {
    // Find the minimum vertex cover in the bipartite graph with the given maximum matching
    // The input is a list of vertices u and v, and a list of edges e
    // tHE OUTPUT IS A LIST OF VERTICES THAT FORM THE MINIMUM VERTEX COVER
    // https://www.slideshare.net/slideshow/ss-86894312/86894312

    let mut not_left_vertex_cover_set = HashSet::new();
    let mut right_vertex_cover_set = HashSet::new();

    // edge set u-> Vec<v>
    let mut u_to_v = HashMap::new();
    let mut v_to_u = HashMap::new();
    for (u_i, v_i) in e.clone() {
        if max_matching.contains_key(&u_i) && max_matching.get(&u_i) == Some(&v_i) {
            v_to_u
                .entry(v_i.clone())
                .or_insert(Vec::new())
                .push(u_i.clone());
        } else {
            u_to_v
                .entry(u_i.clone())
                .or_insert(Vec::new())
                .push(v_i.clone());
        }
    }

    fn traverse_alternating_path(
        u_vertex: &str,
        u_to_v: &HashMap<String, Vec<String>>,
        max_matching: &HashMap<String, String>,
        not_left_vertex_cover_set: &mut HashSet<String>,
        right_vertex_cover_set: &mut HashSet<String>,
    ) {
        not_left_vertex_cover_set.insert(u_vertex.to_string());

        if let Some(v_vertices) = u_to_v.get(u_vertex) {
            for v in v_vertices {
                // Skip if the vertex in V side is already processed
                if right_vertex_cover_set.contains(v) {
                    continue;
                }

                right_vertex_cover_set.insert(v.clone());

                // if there is a vertex in U side that is matched
                if let Some(next_u) = max_matching.get(v) {
                    // if vertex in U side is not processed yet
                    if !not_left_vertex_cover_set.contains(next_u) {
                        traverse_alternating_path(
                            next_u,
                            u_to_v,
                            max_matching,
                            not_left_vertex_cover_set,
                            right_vertex_cover_set,
                        );
                    }
                }
            }
        }
    }

    for u_i in u.clone() {
        if max_matching.contains_key(&u_i) {
            continue;
        }
        traverse_alternating_path(
            &u_i,
            &u_to_v,
            &max_matching,
            &mut not_left_vertex_cover_set,
            &mut right_vertex_cover_set,
        );
    }

    // (u - not_left_vertex_cover_set) | right_vertex_cover_set
    let min_vertex_cover_set: Vec<String> = right_vertex_cover_set
        .union(&u.difference(&not_left_vertex_cover_set).cloned().collect())
        .cloned()
        .collect();
    min_vertex_cover_set
}

#[pyfunction]
fn get_maximal_matching(
    _u: HashSet<String>,
    _e: HashSet<(String, String)>,
) -> HashMap<String, String> {
    todo!()
}

/// A Python module implemented in Rust. The name of this function must match
/// the `lib.name` setting in the `Cargo.toml`, else Python will not be able to
/// import the module.
#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(get_min_vertex_cover, m)?)?;
    m.add_function(wrap_pyfunction!(get_maximal_matching, m)?)?;
    Ok(())
}
