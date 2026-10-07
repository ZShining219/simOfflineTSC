"""Public SUMO report -> static traffic entities (controlled templates v1).

Usage::

    mapper = ReportGrounder(runtime.network_catalog())
    grounded = mapper.map_reports(runtime.reports())

Only public Report fields and a copied static catalog are consumed. Six
active/cleared templates are supported. Arbitrary paraphrases are intentionally
reported as unsupported. Scientific rationale lives in utils.text_grounding.
"""
from dataclasses import replace
import math
import re

from utils.text_grounding import EntityCatalog, GroundedReport, GroundingError, Mention


_ID = r'\S+?'
_LANE = (rf'lane (?P<target_lane>{_ID}) \((?P<movements>[^()]+) lane\), '
         rf'on (?P<direction>eastbound|westbound|northbound|southbound) '
         rf'road (?P<road>{_ID}) approaching junction (?P<approaching>{_ID})')
_ROAD = (rf'directed road (?P<target_edge>{_ID}), from junction (?P<from>{_ID}) '
         rf'to junction (?P<to>{_ID})')
_NUMBER = r'[0-9]+(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?'
_TEMPLATES = (
    ('lane_blockage', 'active', 'lane',
     rf'A stationary obstacle locally blocks {_LANE}, at (?P<position>{_NUMBER}) m from the lane start\.'),
    ('lane_blockage', 'cleared', 'lane', rf'The obstacle on {_LANE} has been removed\.'),
    ('road_closure', 'active', 'edge', rf'All lanes of {_ROAD} are closed to vehicle entry\.'),
    ('road_closure', 'cleared', 'edge',
     rf'The entry restriction imposed by this event on {_ROAD} has been removed\.'),
    ('global_rain', 'active', 'network',
     rf'Rain affects the whole network\. This event limits motor-vehicle lane speeds '
     rf'to (?P<percent>{_NUMBER})% of their normal limits\.'),
    ('global_rain', 'cleared', 'network',
     r'This network-wide rain event has ended and its speed restriction has been removed\.'),
)
# Match whitespace without normalizing input, so character offsets remain valid.
_PATTERNS = tuple((kind, status, scope, re.compile(
    (r'\s*Event (?P<report_id>[A-Za-z0-9_-]+)\. ' + body + r'\s*').replace(' ', r'\s+')))
    for kind, status, scope, body in _TEMPLATES)
_ROLES = {'target_lane': 'lane', 'road': 'edge', 'approaching': 'junction',
          'target_edge': 'edge', 'from': 'junction', 'to': 'junction'}
_MOVEMENTS = {'left-turn', 'through', 'right-turn', 'U-turn'}


class ReportGrounder:
    """Stateless snapshot mapper, strict by default.

    In non-strict mode malformed reports retain their original text and an
    explicit failure status, with no target associations. Downstream consumers
    must check that status. Neither mode consults event IDs to infer targets.
    """

    template_version = 'sumo-report-grounding-v1'

    def __init__(self, catalog, strict=True):
        self.catalog = EntityCatalog(catalog)
        self.strict = strict

    def map_report(self, report):
        result = GroundedReport(report.event_id, report.updated_at, report.text)
        try:
            if (not isinstance(report.text, str) or not isinstance(report.event_id, str)
                    or isinstance(report.updated_at, bool)
                    or not isinstance(report.updated_at, (float, int))
                    or not math.isfinite(report.updated_at) or report.updated_at < 0):
                raise GroundingError('invalid_report', 'Expected text, report ID and finite nonnegative timestamp')
            for kind, status, scope, pattern in _PATTERNS:
                match = pattern.fullmatch(report.text)
                if match is not None:
                    break
            else:
                raise GroundingError('unsupported_template', 'Expected one complete public event report')
            fields = match.groupdict()
            if fields['report_id'] != report.event_id or status != report.status:
                raise GroundingError('metadata_conflict', 'Text ID/status disagrees with public envelope')
            mentions = tuple(Mention(role, entity_type, fields[role], *match.span(role))
                             for role, entity_type in _ROLES.items() if role in fields)
            movement_text = fields.get('movements', '').strip()
            movements = (() if movement_text in ('', 'unspecified-movement') else
                         tuple(part.strip() for part in movement_text.split('/')))
            if not set(movements) <= _MOVEMENTS or len(set(movements)) != len(movements):
                raise GroundingError('invalid_value', 'Unknown or duplicated lane movement')
            percent = float(fields['percent']) if 'percent' in fields else None
            # Runtime's :g formatting can round a factor just below 1 to 100%.
            if percent is not None and not 0 < percent <= 100:
                raise GroundingError('invalid_value', 'Reported rain percentage must be in (0, 100]')
            result = replace(result, event_kind=kind, report_status=status, scope=scope,
                             mentions=mentions, direction=fields.get('direction'), movements=movements,
                             position_m=float(fields['position']) if 'position' in fields else None,
                             speed_percent=percent)
            return self.catalog.link(result)
        except GroundingError as error:
            if self.strict:
                raise
            return replace(result, mapping_status=error.code, diagnostics=(str(error),))

    def map_reports(self, reports):
        """Preserve input order, including cleared reports; empty input -> ().

        Duplicate IDs are invalid even in diagnostic mode: this API accepts a
        latest-report snapshot, not a history containing multiple revisions.
        """
        reports = tuple(reports)
        ids = [r.event_id for r in reports]
        if len(ids) != len(set(ids)):
            raise GroundingError('duplicate_report', 'Expected one latest report per event ID')
        return tuple(self.map_report(report) for report in reports)
