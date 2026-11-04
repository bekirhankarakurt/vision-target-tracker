set(VTT_UNIT_TESTS spectral filters scale kalman control tracking utils)

foreach(name IN LISTS VTT_UNIT_TESTS)
  add_executable(test_${name} test_${name}.cpp)
  target_link_libraries(test_${name} PRIVATE vtt_core)
  vtt_set_project_options(test_${name})
  add_test(NAME unit.${name} COMMAND test_${name})
  set_tests_properties(unit.${name} PROPERTIES TIMEOUT 300)
endforeach()

if(VTT_BUILD_APPS)
  set(SYNTH_VIDEO ${CMAKE_CURRENT_BINARY_DIR}/synthetic.avi)
  set(SYNTH_GT ${CMAKE_CURRENT_BINARY_DIR}/synthetic_gt.csv)

  add_test(NAME integration.generate
           COMMAND vtt_synth --out ${SYNTH_VIDEO} --gt ${SYNTH_GT})
  set_tests_properties(integration.generate PROPERTIES FIXTURES_SETUP synthetic_video)

  foreach(algo mosse kcf)
    add_test(NAME integration.headless_${algo}
             COMMAND vtt --input ${SYNTH_VIDEO} --gt ${SYNTH_GT} --algo ${algo} --headless
                     --csv ${CMAKE_CURRENT_BINARY_DIR}/track_${algo}.csv
                     --min-iou 0.65 --max-center-error 8)
    set_tests_properties(integration.headless_${algo} PROPERTIES
                         FIXTURES_REQUIRED synthetic_video TIMEOUT 300)
  endforeach()

  add_test(NAME integration.overlay_video
           COMMAND vtt --input ${SYNTH_VIDEO} --gt ${SYNTH_GT} --headless --max-frames 30
                   --output ${CMAKE_CURRENT_BINARY_DIR}/overlay.avi)
  set_tests_properties(integration.overlay_video PROPERTIES FIXTURES_REQUIRED synthetic_video)

  add_test(NAME cli.help COMMAND vtt --help)
  add_test(NAME cli.version COMMAND vtt --version)
  add_test(NAME cli.rejects_bad_option COMMAND vtt --bogus)
  set_tests_properties(cli.rejects_bad_option PROPERTIES WILL_FAIL TRUE)
  add_test(NAME cli.rejects_missing_input COMMAND vtt --input does_not_exist.avi --headless --roi 1,1,10,10)
  set_tests_properties(cli.rejects_missing_input PROPERTIES WILL_FAIL TRUE)
  add_test(NAME cli.synth_help COMMAND vtt_synth --help)
endif()

if(VTT_ENABLE_SANITIZERS)
  get_property(vtt_all_tests DIRECTORY PROPERTY TESTS)
  set_tests_properties(${vtt_all_tests} PROPERTIES
    ENVIRONMENT "LSAN_OPTIONS=suppressions=${CMAKE_CURRENT_SOURCE_DIR}/lsan.supp:print_suppressions=0")
endif()
