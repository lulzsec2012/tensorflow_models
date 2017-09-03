; add dc/fm/pt commands/procedures/builtins to typeword and builtin list, 
; dc:syn1312, fm:fm1312-sp2, pt:pt1312, done in Jun. 28, 2014
(defvar tcl-typeword-list
  '("global" "upvar" "inherit" "public" "protected" "private"
    "common" "itk_option" "variable"
    )
  "The part 1 are the original words and part2 are added from dc/pt/fm shells.
List of Tcl keywords denoting \"type\".  Used only for highlighting.
Call `tcl-set-font-lock-keywords' after changing this list.")

(defvar tcl-builtin-list
  '("after" "append" "array" "bgerror" "binary" "catch" "cd" "clock"
    "close" "concat" "console" "dde" "encoding" "eof" "exec" "expr"
    "fblocked" "fconfigure" "fcopy" "file" "fileevent" "flush"
    "format" "gets" "glob" "history" "incr" "info" "interp" "join"
    "lappend" "lindex" "linsert" "list" "llength" "load" "lrange"
    "lreplace" "lsort" "namespace" "open" "package" "pid" "puts" "pwd"
    "read" "regexp" "registry" "regsub" "rename" "scan" "seek" "set"
    "socket" "source" "split" "string" "subst" "tell" "time" "trace"
    "unknown" "unset" "vwait"

    "man" "help" 
    )
  "The part 1 are the original words and part2 are added from dc/pt/fm shells.
List of Tcl commands.  Used only for highlighting.
Call `tcl-set-font-lock-keywords' after changing this list.
This list excludes those commands already found in `tcl-proc-list' and
`tcl-keyword-list'.")

(defvar tcl-keyword-list
  '("if" "then" "else" "elseif" "for" "foreach" "break" "continue" "while"
    "eval" "case" "in" "switch" "default" "exit" "error" "proc" "return"
    "uplevel" "constructor" "destructor" "itcl_class" "loop" "for_array_keys"
    "for_recursive_glob" "for_file" "method" "body" "configbody" "class"
    "chain"

    "alias" "apply" "apropos" "chan" "create_command_group" "date" 
    "define_proc_attributes" "dict" "echo" "error_info" "get_app_var"
    "get_command_option_values" "get_message_ids" "get_message_info"
    "get_unix_variable" "getenv" "group_variable" "is_false" "is_true" 
    "lassign" "lminus" "lrepeat" "lreverse" "ls" "lsearch" "lset" 
    "parse_proc_arguments" "print_message_info" "print_suppressed_messages" 
    "print_variable_group" "printenv" "printvar" "proc_args" "proc_body" 
    "report_app_var" "set_app_var" "set_current_command_mode" "set_message_info"
    "set_unix_variable" "setenv" "sh" "suppress_message" "unalias" "unload" 
    "unsetenv" "unsuppress_message" "update" "which" "write_app_var"

    "::set::highlights" "_version" "acs_check_directories" "acs_compile_design"
    "acs_create_directories" "acs_customize_directory_structure"
    "acs_get_parent_partition" "acs_get_path" "acs_merge_design" "acs_read_hdl"
    "acs_recompile_design" "acs_refine_design" "acs_remove_dont_touch"
    "acs_report_attribute" "acs_report_directories" "acs_report_user_messages"
    "acs_reset_directory_structure" "acs_set_attribute" "acs_submit"
    "acs_submit_large" "acs_write_html" "add_ldb_views" "add_module"
    "add_pg_pin_to_db" "add_pg_pin_to_lib" "add_port_state" "add_power_state"
    "add_pst_state" "add_to_collection" "add_variation" "alib_analyze_libs"
    "all_active_scenarios" "all_clock_gates" "all_clocks" "all_cluster_cells"
    "all_clusters" "all_connected" "all_correlations" "all_critical_cells"
    "all_critical_pins" "all_designs" "all_dont_touch" "all_drc_violated_nets"
    "all_fanin" "all_fanout" "all_high_fanout" "all_ideal_nets" "all_inputs"
    "all_instances" "all_isolation_cells" "all_level_shifters" "all_outputs"
    "all_registers" "all_scenarios" "all_threestate" "all_tieoff_cells"
    "all_upf_repeater_cells" "all_variations" "analyze" "analyze_datapath"
    "analyze_datapath_extraction" "analyze_dw_power" "analyze_minpwr_library"
    "analyze_mv_design" "analyze_points" "append_to_collection"
    "apply_clock_gate_latency" "associate_supply_set" "balance_buffer"
    "balance_registers" "cell_of" "change_link" "change_names"
    "change_selection" "change_selection_no_core"
    "change_selection_too_many_objects" "characterize" "characterize_context"
    "check_bindings" "check_block_abstraction" "check_block_scope" "check_bsd"
    "check_budget" "check_constraints" "check_design" "check_error"
    "check_implementations" "check_isolation_cells" "check_level_shifters"
    "check_library" "check_license" "check_mv_design" "check_noise"
    "check_power" "check_scan_def" "check_synlib" "check_target_library_subset"
    "check_timing" "clean_buffer_tree" "close_mw_lib" "commit_edits"
    "compare_collections" "compare_delay_calculation" "compare_edits"
    "compare_interface_timing" "compare_lib" "compile" "compile_mcl"
    "compile_partitions" "compile_ultra" "complete_net_parasitics"
    "compute_polygons" "connect_logic_net" "connect_net" "connect_pin"
    "connect_supply_net" "convert_db" "convert_from_polygon" "convert_pg"
    "convert_to_polygon" "copy_collection" "copy_design" "copy_mw_lib"
    "cores_used" "cputime" "create_block_abstraction" "create_bsd_patterns"
    "create_bus" "create_cache" "create_cell" "create_clock" "create_cluster"
    "create_constraint_type" "create_container" "create_correlation"
    "create_cutpoint_blackbox" "create_design" "create_dft_netlist"
    "create_generated_clock" "create_ilm" "create_logic_net" "create_logic_port"
    "create_multibit" "create_mw_lib" "create_net" "create_operating_conditions"
    "create_pass_directories" "create_placement_blockage" "create_port"
    "create_power_domain" "create_power_group" "create_power_rail_mapping"
    "create_power_switch" "create_primitive" "create_pst" "create_qtm_clock"
    "create_qtm_constraint_arc" "create_qtm_delay_arc" "create_qtm_drive_type"
    "create_qtm_generated_clock" "create_qtm_insertion_delay"
    "create_qtm_load_type" "create_qtm_model" "create_qtm_path_type"
    "create_qtm_port" "create_scenario" "create_si_context" "create_supply_net"
    "create_supply_port" "create_supply_set" "create_test_protocol"
    "create_variation" "create_voltage_area" "create_wire_load"
    "current_container" "current_design" "current_design_name"
    "current_dft_partition" "current_instance" "current_mw_lib"
    "current_power_rail" "current_prefix" "current_scenario" "current_test_mode"
    "dbatt" "dbatt_dc" "dc_allocate_budgets" "debug_library_cell" "decrypt_lib"
    "decrypt_profile" "define_design_lib" "define_design_mode_group"
    "define_dft_design" "define_dft_partition" "define_name_maps"
    "define_name_rules" "define_primitive_pg_pins" "define_qtm_attribute"
    "define_scaling_lib_group" "define_test_mode" "define_user_attribute"
    "delete_operating_conditions" "derive_clocks" "derive_constraints" "dft_drc"
    "diagnose" "disconnect_net" "drive_of" "duplicate_clock_constraints"
    "duplicate_logic" "elaborate" "elaborate_library_cells" "elapsed_time"
    "enable_primetime_icc_consistency_settings" "enable_write_lib_mode"
    "encrypt_lib" "estimate_clock_network_power" "estimate_eco"
    "extend_mw_layers" "extract_model" "filter" "filter_collection" "find"
    "find_cells" "find_compare_points" "find_designs" "find_drivers"
    "find_equivalent_nets" "find_nets" "find_objects" "find_pins" "find_ports"
    "find_receivers" "find_references" "find_region_of_nets" "find_segments"
    "find_svf_operation" "fix_eco_drc" "fix_eco_leakage" "fix_eco_timing"
    "foreach_in_collection" "fv_svf_processSVP" "generate_eco_map_file"
    "generate_mv_constraints" "get_alternative_lib_cells" "get_always_on_logic"
    "get_attribute" "get_buffers" "get_cell" "get_cells"
    "get_clock_network_objects" "get_clocks" "get_clusters" "get_correlations"
    "get_current_power_domain" "get_current_power_net" "get_design"
    "get_design_lib_path" "get_designs" "get_distributed_variables"
    "get_dont_touch_cells" "get_dont_touch_nets" "get_flat_cells"
    "get_flat_nets" "get_flat_pins" "get_generated_clock" "get_generated_clocks"
    "get_ilm_objects" "get_ilms" "get_latch_loop_groups" "get_lib"
    "get_lib_attribute" "get_lib_cell" "get_lib_cells" "get_lib_pin"
    "get_lib_pins" "get_lib_timing_arcs" "get_libs" "get_license"
    "get_multibits" "get_net" "get_nets" "get_noise_violation_sources"
    "get_object_name" "get_path_group" "get_path_groups" "get_pin" "get_pins"
    "get_polygon_area" "get_port" "get_ports" "get_power_domains"
    "get_power_group_objects" "get_power_switches" "get_qtm_ports"
    "get_random_numbers" "get_references" "get_related_supply_net"
    "get_rp_groups" "get_scan_cells_of_chain" "get_scan_chains"
    "get_scan_chains_by_name" "get_selection" "get_si_bottleneck_nets"
    "get_supply_nets" "get_supply_ports" "get_supply_sets"
    "get_switching_activity" "get_timing_arcs" "get_timing_paths"
    "get_variation_attribute" "get_variations"
    "get_zero_interconnect_delay_mode" "group" "group_path" "gui_bin"
    "gui_change_highlight" "gui_create_attrgroup" "gui_create_pref_category"
    "gui_create_pref_key" "gui_delete_attrgroup" "gui_eval_command"
    "gui_exist_pref_category" "gui_exist_pref_key" "gui_get_current_task"
    "gui_get_highlight" "gui_get_highlight_options" "gui_get_pref_keys"
    "gui_get_pref_value" "gui_get_setting" "gui_get_task_list"
    "gui_get_window_ids" "gui_get_window_pref_categories"
    "gui_get_window_pref_keys" "gui_get_window_pref_value"
    "gui_get_window_types" "gui_list_attrgroups" "gui_remove_pref_key"
    "gui_set_current_task" "gui_set_highlight_options" "gui_set_pref_value"
    "gui_set_setting" "gui_set_window_pref_key" "gui_show_man_page" "gui_start"
    "gui_stop" "gui_update_attrgroup" "gui_update_pref_file" "guide"
    "guide_architecture_db" "guide_architecture_netlist"
    "guide_arithmetic_fracture" "guide_boundary" "guide_boundary_netlist"
    "guide_change_names" "guide_checkpoint" "guide_constraints" "guide_datapath"
    "guide_dont_verify_scan" "guide_dsp_implementation" "guide_dsp_netlist"
    "guide_dsp_pack" "guide_eco_change" "guide_eco_map" "guide_environment"
    "guide_exec" "guide_fsm_reencoding" "guide_group" "guide_group_function"
    "guide_implementation" "guide_info" "guide_instance_map"
    "guide_instance_merging" "guide_inv_push" "guide_mark" "guide_mc"
    "guide_merge" "guide_mim_retiming" "guide_multibit" "guide_multiplier"
    "guide_netlist_table" "guide_pins_eqop" "guide_port_constant"
    "guide_port_punch" "guide_private" "guide_reg_constant"
    "guide_reg_duplication" "guide_reg_encoding" "guide_reg_eqop"
    "guide_reg_merging" "guide_reg_removal" "guide_reg_split"
    "guide_rename_design" "guide_replace" "guide_retiming"
    "guide_retiming_decompose" "guide_retiming_dw_pipeline"
    "guide_retiming_finished" "guide_retiming_multibit" "guide_rewire"
    "guide_scan_input" "guide_scan_output" "guide_sequential_cg_fanin"
    "guide_sequential_cg_fanout" "guide_set_rounding" "guide_share"
    "guide_timebegin" "guide_timeend" "guide_transformation" "guide_ungroup"
    "guide_uniquify" "guide_ununiquify" "guide_upf_copy" "identify_clock_gating"
    "identify_interface_logic" "index_collection" "infer_switching_activity"
    "insert_buffer" "insert_clock_gating" "insert_dft" "insert_inversion"
    "insert_isolation_cell" "insert_level_shifters" "insert_mv_cells"
    "invert_pin" "lib2saif" "library_verification" "license_users" "link"
    "link_design" "list_attributes" "list_designs" "list_dont_touch_types"
    "list_duplicate_designs" "list_files" "list_hdl_blocks" "list_instances"
    "list_key_bindings" "list_libraries" "list_libs" "list_licenses"
    "list_size_only_types" "list_test_models" "list_test_modes" "load_of"
    "load_upf" "map_design_mode" "map_isolation_cell"
    "map_level_shifter_cell" "map_power_switch" "map_retention_cell" "match"
    "max_variation" "mem" "mem_dump_group" "memory" "merge_models" "merge_saif"
    "min_variation" "multicorner_check_cells" "multicorner_is_on" "name_format"
    "open_mw_lib" "optimize_netlist" "optimize_registers" "parallel_execute"
    "parent_cluster" "preview_dft" "print_proc_new_vars" "propagate_constraints"
    "propagate_switching_activity" "push_down_model" "query_cell_instances"
    "query_cell_mapped" "query_map_power_switch" "query_net_ports"
    "query_objects" "query_port_net" "query_port_state" "query_power_switch"
    "query_pst" "query_pst_state" "quit!" "read_aocvm" "read_bsdl"
    "read_container" "read_db" "read_ddc" "read_edif" "read_file"
    "read_fsm_states" "read_lib" "read_milkyway" "read_parasitics"
    "read_partition" "read_pin_map" "read_power_model" "read_saif"
    "read_scan_def" "read_sdc" "read_sdf" "read_sverilog" "read_test_model"
    "read_test_protocol" "read_vcd" "read_verilog" "read_vhdl" "rebuild_mw_lib"
    "record_edits" "redirect" "remote_execute" "remove_annotated_check"
    "remove_annotated_clock_network_power" "remove_annotated_delay"
    "remove_annotated_parasitics" "remove_annotated_power"
    "remove_annotated_transition" "remove_annotations" "remove_aocvm"
    "remove_attribute" "remove_black_box" "remove_boundary_cell"
    "remove_boundary_cell_io" "remove_bsd_ac_port" "remove_bsd_compliance"
    "remove_bsd_instruction" "remove_bsd_linkage_port"
    "remove_bsd_power_up_reset" "remove_buffer" "remove_bus" "remove_cache"
    "remove_capacitance" "remove_case_analysis" "remove_cell"
    "remove_cell_degradation" "remove_clock" "remove_clock_gating"
    "remove_clock_gating_check" "remove_clock_gating_style"
    "remove_clock_groups" "remove_clock_latency" "remove_clock_sense"
    "remove_clock_transition" "remove_clock_uncertainty" "remove_clusters"
    "remove_compare_rules" "remove_connection_class" "remove_constant"
    "remove_constraint" "remove_constraint_type" "remove_container"
    "remove_context" "remove_coupling_separation" "remove_cutpoint"
    "remove_data_check" "remove_design" "remove_design_library"
    "remove_design_mode" "remove_dft_clock_gating_pin" "remove_dft_connect"
    "remove_dft_design" "remove_dft_equivalent_signals" "remove_dft_location"
    "remove_dft_partition" "remove_dft_power_control" "remove_dft_signal"
    "remove_disable_clock_gating_check" "remove_disable_timing"
    "remove_dont_cut" "remove_dont_match_points" "remove_dont_verify_points"
    "remove_dp_int_round" "remove_drive_resistance" "remove_driving_cell"
    "remove_factor_point" "remove_fanout_load" "remove_from_collection"
    "remove_generated_clock" "remove_guidance" "remove_host_options"
    "remove_ideal_latency" "remove_ideal_net" "remove_ideal_network"
    "remove_ideal_transition" "remove_input_delay" "remove_input_noise"
    "remove_input_value_range" "remove_inv_push" "remove_inversion"
    "remove_isolate_ports" "remove_isolation_cell" "remove_ldb_views"
    "remove_level_shifters" "remove_lib" "remove_library" "remove_license"
    "remove_link_library_subset" "remove_max_area" "remove_max_capacitance"
    "remove_max_fanout" "remove_max_time_borrow" "remove_max_transition"
    "remove_min_capacitance" "remove_min_pulse_width" "remove_multibit"
    "remove_net" "remove_noise_immunity_curve" "remove_noise_lib_pin"
    "remove_noise_margin" "remove_object" "remove_operating_conditions"
    "remove_output_delay" "remove_parameters" "remove_parasitic_corner"
    "remove_pass_directories" "remove_path_group" "remove_pin_map"
    "remove_pin_name_synonym" "remove_port" "remove_port_fanout_number"
    "remove_power_domain" "remove_power_groups" "remove_probe_points"
    "remove_propagated_clock" "remove_pulse_clock_max_transition"
    "remove_pulse_clock_max_width" "remove_pulse_clock_min_transition"
    "remove_pulse_clock_min_width" "remove_qtm_attribute" "remove_rail_voltage"
    "remove_resistance" "remove_resistive_drivers" "remove_rtl_load"
    "remove_scaling_lib_group" "remove_scan_group" "remove_scan_link"
    "remove_scan_path" "remove_scan_register_type" "remove_scan_replacement"
    "remove_scan_suppress_toggling" "remove_scenario" "remove_sdc"
    "remove_sense" "remove_setup_hold_pessimism_reduction"
    "remove_si_aggressor_exclusion" "remove_si_delay_analysis"
    "remove_si_delay_disable_statistical" "remove_si_noise_analysis"
    "remove_si_noise_disable_statistical" "remove_steady_state_resistance"
    "remove_target_library_subset" "remove_test_assume" "remove_test_mode"
    "remove_test_model" "remove_test_point_element" "remove_test_power_modes"
    "remove_test_protocol" "remove_unconnected_ports" "remove_upf"
    "remove_user_attribute" "remove_user_budget" "remove_user_match"
    "remove_variation" "remove_verification_priority" "remove_verify_points"
    "remove_wire_load_min_block_size" "remove_wire_load_model"
    "remove_wire_load_selection_group" "rename_cell" "rename_design"
    "rename_mw_lib" "rename_net" "rename_object" "reoptimize_design"
    "replace_clock_gates" "replace_synthetic" "report_aborted_points"
    "report_activity_file_check" "report_activity_waveforms"
    "report_alternative_lib_cells" "report_always_on_cells"
    "report_analysis_coverage" "report_analysis_results"
    "report_annotated_check" "report_annotated_delay"
    "report_annotated_parasitics" "report_annotated_power"
    "report_annotated_transition" "report_aocvm" "report_architecture"
    "report_area" "report_attribute" "report_auto_ungroup"
    "report_autofix_configuration" "report_autofix_element" "report_black_boxes"
    "report_block_abstraction" "report_bottleneck" "report_boundary_cell"
    "report_boundary_cell_io" "report_bsd_ac_port" "report_bsd_buffers"
    "report_bsd_compliance" "report_bsd_configuration" "report_bsd_instruction"
    "report_bsd_linkage_port" "report_bsd_patterns" "report_bsd_power_up_reset"
    "report_budget" "report_buffer_tree" "report_buffer_tree_qor" "report_bus"
    "report_cache" "report_case_analysis" "report_cell" "report_cell_list"
    "report_cell_mode" "report_cell_usage" "report_check_library_options"
    "report_checksum" "report_clock" "report_clock_gate_savings"
    "report_clock_gating" "report_clock_gating_check" "report_clock_timing"
    "report_clock_tree" "report_clocks" "report_clusters" "report_compare_rules"
    "report_compile_options" "report_constant_sources" "report_constants"
    "report_constraint" "report_constraint_type" "report_containers"
    "report_context" "report_crpr" "report_cutpoints" "report_datapath_gating"
    "report_delay_calculation" "report_design" "report_design_lib"
    "report_design_libraries" "report_design_mismatch" "report_designs"
    "report_dft_clock_controller" "report_dft_clock_gating_configuration"
    "report_dft_clock_gating_pin" "report_dft_configuration"
    "report_dft_connect" "report_dft_design" "report_dft_drc_rules"
    "report_dft_drc_violations" "report_dft_equivalent_signals"
    "report_dft_insertion_configuration" "report_dft_location"
    "report_dft_partition" "report_dft_power_control" "report_dft_signal"
    "report_diagnosed_matching_regions" "report_direct_power_rail_tie"
    "report_disable_timing" "report_dont_cuts" "report_dont_match_points"
    "report_dont_touch" "report_dont_verify_points" "report_dp_int_round"
    "report_dp_smartgen_options" "report_driver_model" "report_edits"
    "report_electrical_checks" "report_error_candidates" "report_etm_arc"
    "report_exceptions" "report_factor_points" "report_failing_points"
    "report_fsm" "report_global_slack" "report_global_timing" "report_guidance"
    "report_hdlin_mismatches" "report_hierarchy" "report_host_options"
    "report_host_usage" "report_hybrid_stats" "report_hyperscale"
    "report_ideal_network" "report_ilm" "report_infeasible_paths"
    "report_inferred_opconds" "report_input_value_range"
    "report_interclock_relation" "report_internal_loads" "report_inv_push"
    "report_inversion" "report_isolate_ports" "report_isolation_cell"
    "report_latch_loop_groups" "report_level_shifter" "report_lib"
    "report_lib_groups" "report_libraries" "report_link_library_subset"
    "report_logicbist_configuration" "report_loops" "report_matched_points"
    "report_min_period" "report_min_pulse_width" "report_mode"
    "report_multi_input_switching_coefficient" "report_multibit"
    "report_multidriven_nets" "report_mv_library_cells" "report_mw_lib"
    "report_name_mapping" "report_name_rules" "report_names" "report_net"
    "report_net_fanout" "report_noise" "report_noise_calculation"
    "report_noise_parameters" "report_noise_violation_sources"
    "report_not_compared_points" "report_ocvm" "report_opcond_inference"
    "report_operating_conditions" "report_parameters" "report_partitions"
    "report_pass_data" "report_passing_points" "report_path_budget"
    "report_path_group" "report_pin_map" "report_pin_name_synonym"
    "report_pipeline_scan_data_configuration" "report_port" "report_power"
    "report_power_analysis_options" "report_power_calculation"
    "report_power_derate" "report_power_domain" "report_power_gating"
    "report_power_groups" "report_power_network" "report_power_pin_info"
    "report_power_rail_mapping" "report_power_switch" "report_probe_points"
    "report_probe_status" "report_profile" "report_pst"
    "report_pulse_clock_max_transition" "report_pulse_clock_max_width"
    "report_pulse_clock_min_transition" "report_pulse_clock_min_width"
    "report_qor" "report_qtm_model" "report_reference" "report_related_supplies"
    "report_resources" "report_retention_cell" "report_saif"
    "report_scale_parasitics" "report_scan_cell_set" "report_scan_chain"
    "report_scan_compression_configuration" "report_scan_configuration"
    "report_scan_group" "report_scan_link" "report_scan_path"
    "report_scan_register_type" "report_scan_replacement" "report_scan_state"
    "report_scan_suppress_toggling" "report_scenarios" "report_scope_data"
    "report_sense" "report_serialize_configuration" "report_setup_status"
    "report_si_aggressor_exclusion" "report_si_bottleneck"
    "report_si_delay_analysis" "report_si_double_switching"
    "report_si_noise_analysis" "report_size_only" "report_status"
    "report_streaming_compression_configuration" "report_supply_net"
    "report_supply_port" "report_supply_set" "report_svf" "report_svf_operation"
    "report_switching_activity" "report_synlib" "report_target_library_subset"
    "report_test_assume" "report_test_model" "report_test_point_configuration"
    "report_test_point_element" "report_test_power_modes"
    "report_testability_configuration" "report_threshold_voltage_group"
    "report_timing" "report_timing_derate" "report_timing_requirements"
    "report_top_implementation_options" "report_transitive_fanin"
    "report_transitive_fanout" "report_truth_table" "report_ultra_optimization"
    "report_undriven_nets" "report_units" "report_unmatched_points"
    "report_unread_endpoints" "report_unverified_points" "report_upf"
    "report_use_test_model" "report_user_matches" "report_variation"
    "report_vcd_hierarchy" "report_verify_points" "report_vhdl"
    "report_wire_load" "report_wrapper_configuration" "report_write_lib_mode"
    "reset_aocvm_table_group" "reset_autofix_configuration"
    "reset_autofix_element" "reset_bsd_configuration" "reset_cell_mode"
    "reset_clock_gate_latency" "reset_design" "reset_dft_clock_controller"
    "reset_dft_clock_gating_configuration" "reset_dft_configuration"
    "reset_dft_drc_rules" "reset_dft_insertion_configuration"
    "reset_infeasible_paths" "reset_logicbist_configuration" "reset_mode"
    "reset_multi_input_switching_coefficient" "reset_noise_parameters"
    "reset_ocvm_table_group" "reset_path"
    "reset_pipeline_scan_data_configuration" "reset_power_derate"
    "reset_rtl_to_gate_name" "reset_scale_parasitics"
    "reset_scan_compression_configuration" "reset_scan_configuration"
    "reset_serialize_configuration" "reset_streaming_compression_configuration"
    "reset_switching_activity" "reset_test_mode"
    "reset_test_point_configuration" "reset_testability_configuration"
    "reset_timing_derate" "reset_variation" "reset_wrapper_configuration"
    "resize_polygon" "restore_session" "rewire_clock_gating" "rewire_connection"
    "saif_map" "save_qtm_model" "save_session" "save_upf" "scale_parasitics"
    "select_cell_list" "set::add" "set::contains" "set::create"
    "set::create_empty_set" "set::difference" "set::fanin" "set::fanout"
    "set::from_find_equivalent_nets" "set::get" "set::intersection" "set::load"
    "set::parents" "set::remove" "set::reset_highlights" "set::save"
    "set::translate" "set::union" "set::validate" "set_active_clocks"
    "set_active_scenarios" "set_always_on_cell" "set_always_on_strategy"
    "set_annotated_check" "set_annotated_clock_network_power"
    "set_annotated_delay" "set_annotated_power" "set_annotated_transition"
    "set_aocvm_coefficient" "set_aocvm_table_group" "set_architecture"
    "set_attribute" "set_auto_disable_drc_nets" "set_auto_ideal_nets"
    "set_autofix_configuration" "set_autofix_element" "set_balance_registers"
    "set_black_box" "set_boundary_cell" "set_boundary_cell_io"
    "set_boundary_optimization" "set_bsd_ac_port" "set_bsd_compliance"
    "set_bsd_configuration" "set_bsd_instruction" "set_bsd_linkage_port"
    "set_bsd_power_up_reset" "set_case_analysis" "set_cell_degradation"
    "set_cell_internal_power" "set_cell_mode" "set_check_library_options"
    "set_cle_options" "set_clock" "set_clock_gate_latency"
    "set_clock_gating_check" "set_clock_gating_enable"
    "set_clock_gating_objects" "set_clock_gating_registers"
    "set_clock_gating_style" "set_clock_groups" "set_clock_latency"
    "set_clock_map" "set_clock_sense" "set_clock_skew" "set_clock_transition"
    "set_clock_uncertainty" "set_combinational_type" "set_compare_point"
    "set_compare_rule" "set_compile_directives" "set_compile_partitions"
    "set_connection_class" "set_constant" "set_constraint" "set_context_margin"
    "set_cost_priority" "set_coupling_separation" "set_critical_range"
    "set_cross_voltage_domain_analysis_guardband" "set_current_power_domain"
    "set_current_power_net" "set_cutpoint" "set_data_check"
    "set_datapath_gating_options" "set_datapath_optimization_effort"
    "set_default_drive" "set_default_driving_cell" "set_default_fanout_load"
    "set_default_input_delay" "set_default_load" "set_default_output_delay"
    "set_delay_calculation" "set_design_attributes" "set_design_license"
    "set_design_top" "set_dft_clock_controller"
    "set_dft_clock_gating_configuration" "set_dft_clock_gating_pin"
    "set_dft_configuration" "set_dft_connect" "set_dft_drc_configuration"
    "set_dft_drc_rules" "set_dft_equivalent_signals"
    "set_dft_insertion_configuration" "set_dft_location" "set_dft_power_control"
    "set_dft_signal" "set_direct_power_rail_tie" "set_direction"
    "set_disable_clock_gating_check" "set_disable_timing"
    "set_distributed_parameters" "set_distributed_variables"
    "set_domain_supply_net" "set_dont_cut" "set_dont_match_points"
    "set_dont_override" "set_dont_retime" "set_dont_touch"
    "set_dont_touch_network" "set_dont_use" "set_dont_verify_points"
    "set_dp_int_round" "set_dp_smartgen_options" "set_drive"
    "set_drive_resistance" "set_driving_cell" "set_dynamic_optimization"
    "set_eco_options" "set_equal" "set_factor_point" "set_false_path"
    "set_fanout_load" "set_fix_hold" "set_fix_multiple_port_nets" "set_flatten"
    "set_fsm_encoding" "set_fsm_encoding_style" "set_fsm_minimize"
    "set_fsm_order" "set_fsm_preserve_state" "set_fsm_state_vector"
    "set_fuzzy_query_options" "set_host_options" "set_hyperscale_config"
    "set_hyperscale_context_margin" "set_hyperscale_eco_context"
    "set_ideal_latency" "set_ideal_net" "set_ideal_network"
    "set_ideal_transition" "set_impl_priority" "set_implementation"
    "set_implementation_design" "set_input_delay" "set_input_noise"
    "set_input_transition" "set_input_value_range" "set_inv_push"
    "set_isolate_ports" "set_isolation" "set_isolation_cell"
    "set_isolation_control" "set_latch_loop_breaker" "set_latch_loop_breakers"
    "set_lbist_configuration" "set_leakage_optimization"
    "set_leakage_power_model" "set_level_shifter" "set_level_shifter_cell"
    "set_level_shifter_strategy" "set_level_shifter_threshold"
    "set_lib_attribute" "set_lib_rail_connection" "set_libcell_dimensions"
    "set_libpin_location" "set_library_driver_waveform"
    "set_link_library_subset" "set_load" "set_local_link_library" "set_logic_dc"
    "set_logic_one" "set_logic_zero" "set_logicbist_configuration"
    "set_map_only" "set_max_area" "set_max_capacitance" "set_max_delay"
    "set_max_dynamic_power" "set_max_fanout" "set_max_leakage_power"
    "set_max_net_length" "set_max_time_borrow" "set_max_transition"
    "set_message_severity" "set_min_capacitance" "set_min_delay"
    "set_min_library" "set_min_pulse_width" "set_minimize_tree_delay" "set_mode"
    "set_model_drive" "set_model_load" "set_model_map_effort"
    "set_multi_input_switching_coefficient" "set_multi_vth_constraint"
    "set_multibit_options" "set_multicycle_path" "set_mw_lib_reference"
    "set_mw_technology_file" "set_noise_derate" "set_noise_immunity_curve"
    "set_noise_lib_pin" "set_noise_margin" "set_noise_parameters"
    "set_ocvm_table_group" "set_opcond_inference" "set_operating_conditions"
    "set_opposite" "set_optimize_registers" "set_output_clock_port_type"
    "set_output_delay" "set_parameters" "set_parasitic_corner"
    "set_partial_on_translation" "set_path_margin" "set_pg_pin_model"
    "set_pin_model" "set_pin_name_synonym"
    "set_pipeline_scan_data_configuration" "set_port_abstraction"
    "set_port_attributes" "set_port_fanout_number" "set_port_location"
    "set_power_analysis_options" "set_power_clock_scaling" "set_power_derate"
    "set_power_gating_style" "set_power_switch_cell" "set_prefer"
    "set_preserve_clock_gate" "set_probe_points" "set_program_options"
    "set_propagated_clock" "set_pulse_clock_cell"
    "set_pulse_clock_max_transition" "set_pulse_clock_max_width"
    "set_pulse_clock_min_transition" "set_pulse_clock_min_width"
    "set_qtm_attribute" "set_qtm_global_parameter" "set_qtm_port_drive"
    "set_qtm_port_load" "set_qtm_technology" "set_query_rules"
    "set_rail_voltage" "set_reference_design" "set_register_merging"
    "set_register_replication" "set_register_type" "set_related_supply_net"
    "set_replace_clock_gates" "set_resistance" "set_resource_allocation"
    "set_retention" "set_retention_cell" "set_retention_control"
    "set_retention_control_pins" "set_rtl_load" "set_rtl_to_gate_name"
    "set_scaling_lib_group" "set_scan_compression_configuration"
    "set_scan_configuration" "set_scan_element" "set_scan_group" "set_scan_link"
    "set_scan_path" "set_scan_register_type" "set_scan_replacement"
    "set_scan_state" "set_scan_suppress_toggling" "set_scope"
    "set_script_runtime_report_mode" "set_sense" "set_serialize_configuration"
    "set_setup_hold_pessimism_reduction" "set_si_aggressor_exclusion"
    "set_si_delay_analysis" "set_si_delay_disable_statistical"
    "set_si_noise_analysis" "set_si_noise_disable_statistical" "set_size_only"
    "set_steady_state_resistance" "set_streaming_compression_configuration"
    "set_structure" "set_svf" "set_svf_retiming" "set_switching_activity"
    "set_switching_activity_profile" "set_synlib_dont_get_license"
    "set_tap_elements" "set_target_library_subset" "set_temperature"
    "set_test_assume" "set_test_point_configuration" "set_test_point_element"
    "set_test_power_modes" "set_testability_configuration" "set_timing_derate"
    "set_timing_ranges" "set_top" "set_top_implementation_options"
    "set_transform_for_retiming" "set_ultra_optimization" "set_unconnected"
    "set_ungroup" "set_units" "set_user_attribute" "set_user_budget"
    "set_user_match" "set_variation" "set_variation_correlation"
    "set_variation_library" "set_variation_quantile" "set_verification_priority"
    "set_verify_points" "set_voltage" "set_voltage_model" "set_vsdc"
    "set_wire_load" "set_wire_load_min_block_size" "set_wire_load_mode"
    "set_wire_load_model" "set_wire_load_selection_group"
    "set_wrapper_configuration" "set_zero_interconnect_delay_mode" "setup"
    "sh_list_key_bindings" "share_operations_on_one_resource"
    "shell_is_in_exploration_mode" "shell_is_in_topographical_mode"
    "shell_is_in_xg_mode" "show_cng" "show_cng_node" "show_cng_stats"
    "show_pin_slack" "show_redundant_cng_nodes" "sim_analyze_clock_network"
    "sim_setup_library" "sim_setup_simulator" "sim_setup_spice_deck"
    "sim_validate_setup" "simplify_constants" "size_cell" "sizeof_collection"
    "sort_collection" "start_gui" "start_profile" "stop_gui" "stop_profile"
    "sub_designs_of" "sub_instances_of" "sub_variation" "swap_cell"
    "test_compare_rule" "transform_exceptions" "translate"
    "translate_instance_pathname" "undo_edits" "undo_match" "ungroup" "uniquify"
    "unset_rtl_to_gate_name" "update_lib" "update_lib_model"
    "update_lib_pg_pin_model" "update_lib_pin_model" "update_lib_voltage_model"
    "update_noise" "update_power" "update_scope_data" "update_timing"
    "upf_version" "use_test_model" "variation_correlation" "verify" "write"
    "write_activity_waveforms" "write_arrival_annotations" "write_binary_aocvm"
    "write_bsd_rtl" "write_bsdl" "write_changes" "write_compile_script"
    "write_container" "write_context" "write_design_lib_paths" "write_edits"
    "write_environment" "write_file" "write_hierarchical_verification_script"
    "write_ilm_netlist" "write_ilm_parasitics" "write_ilm_script"
    "write_ilm_sdf" "write_interface_timing" "write_lib"
    "write_lib_specification_model" "write_library_debug_scripts"
    "write_link_library" "write_makefile" "write_milkyway" "write_mw_lib_files"
    "write_parasitics" "write_partition" "write_partition_constraints"
    "write_physical_annotations" "write_power_model" "write_profile"
    "write_qtm_model" "write_rtl_load" "write_saif" "write_scan_def"
    "write_script" "write_sdc" "write_sdf" "write_sdf_constraints"
    "write_spice_deck" "write_test" "write_test_model" "write_test_protocol"
    "write_tmax_library"
    )
  "List of Tcl keywords.  Used only for highlighting.
Default list includes some TclX keywords.
Call `tcl-set-font-lock-keywords' after changing this list.")