# Common Arguments

There are a few arguments that show up in many expressions.
Rather than repeating them everywhere (and risking doc drift), they are all described here.

## `on_row_error`

Some kernels can error depending on the data in a row.
Something could be out of bound, or a line might cross itself where an expression requires it not to.
With the `on_row_error` variable, you can set what to do if that happens.

- `on_row_error="null"` instructs the kernel to simply mark the resulting row as null (invalid).
  The total rows coming out of the expression then does not change.
- `on_row_error="drop"` tells it to simply omit that row from the result.
  The amount of rows in the output can be the same if no errors occurred, or lower if they did.
