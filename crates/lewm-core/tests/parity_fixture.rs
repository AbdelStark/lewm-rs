//! RFC 0008 parity fixture contract tests.

mod support;

#[test]
fn parity_fixture_loads_with_expected_shape_and_metadata() {
    let fixture = support::load_fixture().expect("parity fixture should load");
    let meta = support::load_fixture_meta().expect("parity fixture metadata should load");

    assert_eq!(fixture.pixels.shape, [4, 4, 3, 224, 224]);
    assert_eq!(fixture.actions.shape, [4, 4, 2]);
    assert_eq!(fixture.pixels.values.len(), 4 * 4 * 3 * 224 * 224);
    assert_eq!(fixture.actions.values.len(), 4 * 4 * 2);
    assert_eq!(fixture.seed, 0);
    assert_eq!(meta["fixture_seed"], 0);
    assert_eq!(meta["fixture_hash"], fixture.fixture_hash);
    assert_eq!(meta["fixture_hash_algorithm"], "blake3");
    assert_eq!(meta["git_short_sha"], fixture.git_short_sha);
    assert!(
        meta["regeneration_policy"]
            .as_str()
            .expect("regeneration policy should be a string")
            .contains("RFC0008-006")
    );
    assert!(
        fixture
            .pixels
            .values
            .iter()
            .all(|value| value.is_finite() && (-1.0..=1.0).contains(value))
    );
    assert!(fixture.actions.values.iter().all(|value| value.is_finite()));
}
