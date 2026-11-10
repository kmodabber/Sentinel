import random
import unittest
from datetime import datetime, timezone
from unittest.mock import MagicMock

from fake_detections import MODEL_VERSION, generate_event, insert_event, register_model


class FakeDetectionsTests(unittest.TestCase):
    def test_benign_event_matches_c2_constraints(self):
        e = generate_event(random.Random(1), datetime.now(timezone.utc), 0)
        self.assertEqual(e["predicted_class"], "Benign")
        self.assertEqual(e["source"], "replay")
        self.assertFalse(e["is_attack"])
        self.assertTrue(0 <= e["confidence"] <= 1)
        self.assertTrue(0 <= e["src_port"] <= 65535)
        self.assertTrue(e["packets"] >= 0 and e["bytes"] >= 0)

    def test_attack_event_has_detection_fields(self):
        e = generate_event(random.Random(5), attack_probability=1)
        self.assertTrue(e["is_attack"])
        self.assertNotEqual(e["predicted_class"], "Benign")
        self.assertTrue(0 <= e["confidence"] <= 1)
        self.assertIsInstance(e["is_alert"], bool)

    def test_model_version_registered_before_fk_use(self):
        conn = MagicMock()
        register_model(conn)
        sql, values = conn.execute.call_args.args
        self.assertIn("ON CONFLICT", sql)
        self.assertEqual(values[0], MODEL_VERSION)

    def test_flow_and_detection_are_in_same_transaction(self):
        conn = MagicMock()
        conn.execute.return_value.fetchone.return_value = [42]
        e = generate_event(random.Random(7), attack_probability=1)
        flow_id, detected = insert_event(conn, e)
        self.assertEqual(flow_id, 42)
        self.assertTrue(detected)
        self.assertEqual(conn.execute.call_count, 2)
        conn.transaction.assert_called_once()

    def test_benign_inserts_only_flow(self):
        conn = MagicMock()
        conn.execute.return_value.fetchone.return_value = [43]
        e = generate_event(random.Random(9), attack_probability=0)
        _, detected = insert_event(conn, e)
        self.assertFalse(detected)
        self.assertEqual(conn.execute.call_count, 1)

if __name__ == "__main__":
    unittest.main()
