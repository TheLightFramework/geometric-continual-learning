"""CPU-only fault-injection tests; fixtures are not language-model evidence."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from gcl.transactions import Store, artifact, read


class Transactions(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.parent_file = self.root / 'parent.bin'
        self.parent_file.write_bytes(b'immutable parent tensor fixture')
        self.store = Store(self.root / 'store')
        self.parent = self.store.initialize([artifact(self.parent_file)], 'fixture')
        self.original = artifact(self.parent_file)

    def tearDown(self): self.temp.cleanup()

    def candidate(self, passed=True):
        checkpoint = self.root / 'clone.bin'; checkpoint.write_bytes(b'new candidate fixture')
        contract = self.root / 'contract.json'
        contract.write_text(json.dumps({'required_tests': ['acquisition', 'preservation', 'integrity'], 'evidence_standing': 'engineering fixture, not scientific evidence'}))
        artifacts = [artifact(checkpoint)]
        receipt = self.root / 'evaluation.json'
        receipt.write_text(json.dumps({'parent': self.parent, 'artifacts': artifacts, 'contract_sha256': artifact(contract)['sha256'], 'tests': {'acquisition': passed, 'preservation': True, 'integrity': True}}))
        return {'parent': self.parent, 'artifacts': artifacts, 'contract': artifact(contract), 'evaluation': artifact(receipt)}

    def test_reject_leaves_parent(self):
        result = self.store.review(self.candidate(False), apply=True)
        self.assertEqual(result['failed'], ['acquisition'])
        self.assertEqual(self.store.current(), self.parent)
        self.assertEqual(artifact(self.parent_file), self.original)

    def test_preview_does_not_apply(self):
        self.assertTrue(self.store.review(self.candidate())['eligible'])
        self.assertEqual(self.store.current(), self.parent)

    def test_apply_then_exact_rollback(self):
        self.store.review(self.candidate(), apply=True)
        self.assertNotEqual(self.store.current(), self.parent)
        self.assertEqual(self.store.rollback(self.parent), self.parent)
        self.assertEqual(artifact(self.parent_file), self.original)

    def test_stale_parent(self):
        candidate = self.candidate(); self.store.review(candidate, apply=True)
        with self.assertRaisesRegex(ValueError, 'STALE_PARENT'): self.store.review(candidate, apply=True)

    def test_mutated_checkpoint(self):
        candidate = self.candidate(); Path(candidate['artifacts'][0]['path']).write_bytes(b'corrupted')
        with self.assertRaisesRegex(ValueError, 'ARTIFACT_CHANGED'): self.store.review(candidate, apply=True)
        self.assertEqual(self.store.current(), self.parent)

    def test_missing_test(self):
        candidate = self.candidate(); p = Path(candidate['evaluation']['path']); r = read(p)
        del r['tests']['preservation']; p.write_text(json.dumps(r)); candidate['evaluation'] = artifact(p)
        with self.assertRaisesRegex(ValueError, 'MISSING_OR_UNTYPED'): self.store.review(candidate)

    def test_truthy_string_is_not_pass(self):
        candidate = self.candidate(); p = Path(candidate['evaluation']['path']); r = read(p)
        r['tests']['acquisition'] = 'false'; p.write_text(json.dumps(r)); candidate['evaluation'] = artifact(p)
        with self.assertRaisesRegex(ValueError, 'MISSING_OR_UNTYPED'): self.store.review(candidate)

    def test_wrong_receipt_binding(self):
        candidate = self.candidate(); p = Path(candidate['evaluation']['path']); r = read(p)
        r['artifacts'] = [self.original]; p.write_text(json.dumps(r)); candidate['evaluation'] = artifact(p)
        with self.assertRaisesRegex(ValueError, 'EVALUATION_NOT_BOUND'): self.store.review(candidate)

    def test_concurrent_writer_rejected(self):
        candidate = self.candidate()
        with self.store.writer():
            with self.assertRaises(FileExistsError): self.store.review(candidate)

    def test_pointer_failure_keeps_parent(self):
        candidate = self.candidate()
        with patch('gcl.transactions.os.replace', side_effect=OSError('injected crash')):
            with self.assertRaises(OSError): self.store.review(candidate, apply=True)
        self.assertEqual(self.store.current(), self.parent)
        self.assertEqual(artifact(self.parent_file), self.original)

    def test_manifest_corruption_rejected(self):
        p = self.store.root / 'versions' / (self.parent + '.json')
        r = read(p); r['label'] = 'changed'; p.write_text(json.dumps(r))
        with self.assertRaisesRegex(ValueError, 'VERSION_MANIFEST_CHANGED'): self.store.current()

    def test_invalid_rollback_path_rejected(self):
        with self.assertRaises(ValueError): self.store.rollback('../parent')

    def test_evidence_mutation_after_apply_detected(self):
        candidate = self.candidate(); self.store.review(candidate, apply=True)
        Path(candidate['evaluation']['path']).write_text('{}')
        with self.assertRaisesRegex(ValueError, 'ARTIFACT_CHANGED'): self.store.current()

    def test_empty_gate_not_vacuous_success(self):
        candidate = self.candidate(); p = Path(candidate['contract']['path']); r = read(p)
        r['required_tests'] = []; p.write_text(json.dumps(r)); candidate['contract'] = artifact(p)
        with self.assertRaisesRegex(ValueError, 'INVALID_REQUIRED_TESTS'): self.store.review(candidate)


if __name__ == '__main__': unittest.main(verbosity=2)
