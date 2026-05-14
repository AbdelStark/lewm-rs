//! RFC 0008 parity fixture contract tests.

mod support;

#[test]
fn parity_fixture_loads_with_expected_shape_and_metadata() {
    let fixture = support::load_fixture().expect("parity fixture should load");
    let meta = support::load_fixture_meta().expect("parity fixture metadata should load");

    assert_eq!(fixture.pixels.shape, [4, 4, 3, 224, 224]);
    assert_eq!(fixture.actions.shape, [4, 4, 10]);
    assert_eq!(fixture.pixels.values.len(), 4 * 4 * 3 * 224 * 224);
    assert_eq!(fixture.actions.values.len(), 4 * 4 * 10);
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

#[test]
fn reference_model_metadata_locks_pusht_architecture() {
    let meta = support::load_reference_model_meta().expect("reference metadata should load");

    assert_eq!(meta["source_model"]["repo_id"], "quentinll/lewm-pusht");
    assert_eq!(
        meta["source_model"]["revision"],
        "22b330c28c27ead4bfd1888615af1340e3fe9052"
    );
    assert_eq!(
        meta["source_model"]["weights_sha256"],
        "48938400ae3464c9680731287f583a9cb516f55a8ec64ea13a91be47fb15b607"
    );
    assert_eq!(meta["source_model"]["state_dict_tensor_count"], 303);
    assert_eq!(meta["source_model"]["state_dict_value_count"], 18_042_672);

    let arch = &meta["locked_architecture"];
    assert_eq!(arch["encoder"]["size"], "tiny");
    assert_eq!(arch["encoder"]["patch_size"], 14);
    assert_eq!(arch["encoder"]["hidden_size"], 192);
    assert_eq!(arch["encoder"]["num_attention_heads"], 3);
    assert_eq!(arch["action_encoder"]["raw_action_dim"], 2);
    assert_eq!(arch["action_encoder"]["frameskip"], 5);
    assert_eq!(arch["action_encoder"]["input_dim"], 10);
    assert_eq!(arch["action_encoder"]["emb_dim"], 192);
    assert_eq!(arch["predictor"]["num_frames"], 3);
    assert_eq!(arch["predictor"]["heads"], 16);
    assert_eq!(arch["predictor"]["dim_head"], 64);
    assert_eq!(arch["predictor"]["attention_inner_dim"], 1024);
    assert_eq!(arch["projector"]["hidden_dim"], 2048);
    assert_eq!(arch["pred_proj"]["output_dim"], 192);
}
