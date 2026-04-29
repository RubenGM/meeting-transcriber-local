import unittest

from meeting_transcriber.time_range import (
    format_optional_range,
    hms_to_seconds,
    parse_optional_timestamp,
    validate_time_range,
)


class TimeRangeTests(unittest.TestCase):
    def test_parse_optional_timestamp_accepts_blank_as_none(self):
        self.assertIsNone(parse_optional_timestamp(""))
        self.assertIsNone(parse_optional_timestamp("   "))

    def test_parse_optional_timestamp_accepts_seconds_minutes_and_hours(self):
        self.assertEqual(parse_optional_timestamp("90"), 90.0)
        self.assertEqual(parse_optional_timestamp("01:30"), 90.0)
        self.assertEqual(parse_optional_timestamp("01:02:03"), 3723.0)

    def test_parse_optional_timestamp_rejects_bad_values(self):
        with self.assertRaises(ValueError):
            parse_optional_timestamp("1:2:3:4")
        with self.assertRaises(ValueError):
            parse_optional_timestamp("-1")

    def test_validate_time_range_rejects_end_before_start(self):
        with self.assertRaises(ValueError):
            validate_time_range(120.0, 60.0)

    def test_validate_time_range_accepts_open_ranges(self):
        validate_time_range(None, None)
        validate_time_range(60.0, None)
        validate_time_range(None, 120.0)

    def test_hms_to_seconds_treats_all_zero_as_none_when_blank_allowed(self):
        self.assertIsNone(hms_to_seconds("0", "0", "0", blank_when_zero=True))
        self.assertEqual(hms_to_seconds("1", "2", "3", blank_when_zero=True), 3723.0)

    def test_hms_to_seconds_rejects_bad_numeric_input(self):
        with self.assertRaises(ValueError):
            hms_to_seconds("a", "0", "0", blank_when_zero=True)

    def test_format_optional_range(self):
        self.assertEqual(format_optional_range(None, None), "Todo el archivo")
        self.assertEqual(format_optional_range(60, 120), "00:01:00 -> 00:02:00")


if __name__ == "__main__":
    unittest.main()
