"""Strict lower-triangular evaluation matrix helpers and validator."""


def expected_cells(stage_index, networks):
    stage_index = int(stage_index)
    if stage_index < 1 or stage_index > len(networks):
        raise ValueError('stage_index outside network order')
    return tuple(networks[:stage_index])


def validate_lower_triangle(cells, networks, stage_index):
    """Reject future scene evaluations and missing visible scenes."""
    expected = set(expected_cells(stage_index, networks))
    actual = set(cells)
    future = actual - expected
    missing = expected - actual
    if future or missing:
        raise ValueError(
            f'lower-triangle violation: missing={sorted(missing)}, '
            f'future={sorted(future)}'
        )
    return {'valid': True, 'stage_index': int(stage_index),
            'evaluated': list(networks[:stage_index]), 'future': list(networks[stage_index:])}

