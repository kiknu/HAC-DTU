import mitsuba as _mi

from typing import cast, Any

# This patches Mitsuba to work slightly better with static type checking tools
# Without this, Pylance (and others, I expect) paints everything red :(
mi = cast(Any, _mi)  # type: ignore
