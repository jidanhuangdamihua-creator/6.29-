# D2 Source Calendarization Design

## Scope

Implement the frozen D2 source-calendarization rule only. The change applies to
the source frame after the frozen source candidate keys have been selected and
before source completeness eligibility or KNN preparation. It does not change
the target, validation, blind-period frames, or any date window.

## Frozen contract

- Source interval: `2018-01-02..2018-06-30`, exactly 180 Gregorian calendar days.
- Allowed missing dates: `2018-04-01`, `2018-04-25`, `2018-05-01`, and `2018-06-02`.
- Synthetic sales: numeric `0.0` only.
- Entity identity: copied from the current source entity row.
- Calendar fields: regenerated from the inserted row's actual date.
- Other missing non-static fields are never inferred; if a downstream consumer
  requires them, its existing completeness validation must fail closed.
- Any missing date outside the four-date allowlist, duplicate entity/date key,
  missing candidate entity, invalid date, or non-finite sales is a protocol
  violation.

## Data flow

`configure_protocol_frames` will freeze the D2 candidate keys, slice the source
frame to the fixed 180-day interval, then call the D2 calendarizer. The returned
frame is the only frame used to build `PreparedDailySequencePool` and perform
KNN. The calendarizer receives no target or blind-period sales frame.

The calendarizer emits immutable report metadata and attaches it to the source
frame. The source authority digest covers the calendarized source and the rule
version. The consumer-frame fingerprint covers the exact frame handed to
source completeness/KNN. The shared candidate-pool digest includes both values
and the rule version. After selection, the selector derives the final sealed
identity from those values plus the candidate and selection-result digests.

## Failure behavior

The implementation raises `ProtocolViolation` before KNN when the source cannot
be proven to contain exactly the frozen 180 dates for every frozen candidate.
No fallback, fill method, target read, or runtime repair of any other date is
allowed.

## Verification

Focused tests will prove the four synthetic dates, exact 180-day coverage,
date-derived calendar fields, source-only isolation, rejection of other missing
dates and duplicate keys, prohibition of sales interpolation/fill methods, and
digest cascade changes when the rule version or calendarized result changes.
