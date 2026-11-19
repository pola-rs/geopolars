# Winding order

GeoPolars does not really care about winding order. It will take any,
and keep them in their orientation if that means less copying around is required.
This philosophy is extended to external tools.
If we don't need to transform any data, we'll just keep it in the same order.
