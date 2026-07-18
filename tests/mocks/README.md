# MSI test fixtures

`msi/mouse_urinary_bladder_mock.imzML` and its matching `.ibd` file contain six
unchanged spectra from `mouse_urinary_bladder.imzML`. The retained source
spectrum indices are `0`, `1`, `2`, `260`, `261`, and `262`, forming a 3×2
coordinate region.

The original image is intentionally not committed because its `.ibd` file is
approximately 815 MB. Regenerate the compact fixture from a local copy with:

```bash
python tests/mocks/build_msi_fixture.py \
  data/bladder_data/bladder_data/mouse_urinary_bladder.imzML \
  tests/mocks/msi/mouse_urinary_bladder_mock.imzML
```

Reusable reader, active-context, and dataset doubles are defined in
`components.py`. Functional test directories import those objects instead of
building independent mocks.
