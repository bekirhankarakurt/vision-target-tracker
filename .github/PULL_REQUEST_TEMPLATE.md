## Summary

<!-- What does this change and why? -->

## Testing

- [ ] `cmake --build build && ctest --test-dir build` passes
- [ ] No new warnings with `-DVTT_WARNINGS_AS_ERRORS=ON`
- [ ] Accuracy / latency impact checked with `vtt --headless --gt` on the synthetic sequence (if tracker code changed)

## Notes

<!-- Benchmarks, screenshots, follow-ups -->
