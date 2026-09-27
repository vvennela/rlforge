"""Controller-only answer comparison, shared by audit and exported runtime."""
import json
import math
from decimal import Decimal, InvalidOperation


def canonical(value):
    return json.dumps(value, sort_keys=True, allow_nan=False, separators=(',', ':'))


def accepts(row, answer):
    try:
        expected = row['reference_answer']
        if row['verification'] == 'numeric':
            if type(answer) not in (int, float) or not math.isfinite(answer):
                return False
            # Decimal subtraction avoids accepting adjacent large integers after float rounding.
            return abs(Decimal(str(answer)) - Decimal(str(expected))) <= Decimal(str(row.get('tolerance', 0)))
        return canonical(answer) == canonical(expected)
    except (ValueError, TypeError, OverflowError, InvalidOperation):
        return False
