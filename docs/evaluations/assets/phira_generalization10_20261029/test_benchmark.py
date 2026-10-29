"""Reference-clock and score semantics checks for the new community adapter."""
import copy
import importlib.util
from pathlib import Path
import unittest

path = Path(__file__).with_name('benchmark.py')
spec = importlib.util.spec_from_file_location('community_benchmark', path)
benchmark = importlib.util.module_from_spec(spec)
spec.loader.exec_module(benchmark)


class CommunityBenchmarkTests(unittest.TestCase):
    def chart(self):
        return dict(BPMList=[dict(bpm=120, startTime=[0, 0, 1])], META=dict(offset=-100),
            judgeLineList=[dict(notes=[dict(type=kind, startTime=[i+2, 1, 2],
                endTime=[i+3, 1, 2], positionX=0, isFake=0)
                for i, kind in enumerate((1, 2, 3, 4))])])

    def test_original_clock_types_and_fake_notes(self):
        chart=self.chart()
        fake=copy.deepcopy(chart['judgeLineList'][0]['notes'][0])
        fake['isFake']=1; fake['alpha']=0
        chart['judgeLineList'][0]['notes'].append(fake)
        notes,clock,screen=benchmark.reference_rpe(chart,dict(offset=.04))
        self.assertAlmostEqual(clock['beat_zero_seconds'], -.06)
        self.assertAlmostEqual(notes[0]['time_seconds'], 1.19)
        self.assertEqual([n['type'] for n in notes], ['tap','hold','flick','drag'])
        self.assertAlmostEqual(notes[1]['end_seconds']-notes[1]['time_seconds'], .5)
        self.assertEqual(notes[0]['time_seconds'], notes[0]['end_seconds'])
        self.assertEqual(screen['decorative_fake_notes_excluded'], 1)

    def test_reject_unsupported_clock_and_hidden_judgment(self):
        chart=self.chart()
        chart['BPMList'].append(dict(bpm=150,startTime=[8,0,1]))
        with self.assertRaises(ValueError):benchmark.reference_rpe(chart,{})
        chart=self.chart();chart['judgeLineList'][0]['notes'][0]['alpha']=0
        with self.assertRaises(ValueError):benchmark.reference_rpe(chart,{})

    def test_reference_agreement_is_distinct_from_completion(self):
        notes,clock,_=benchmark.reference_rpe(self.chart(),{})
        prediction=dict(status='completed',notes=[dict(n,position=0,screen_x=.3) for n in notes])
        profile=benchmark.read(benchmark.ROOT/'benchmarks/v1/protocol.json')['rhythm']
        score=benchmark.score(notes,prediction,clock,profile)
        self.assertEqual(score['onset_f1_50ms'],1)
        self.assertEqual(score['time_type_f1'],1)
        prediction['status']='failed'
        failed=benchmark.score(notes,prediction,clock,profile)
        self.assertEqual(failed['time_type_f1'],1)
        self.assertEqual(failed['completed_only_time_type_f1'],0)
        for n in prediction['notes']:n['time_seconds']+=.2;n['end_seconds']+=.2
        shifted=benchmark.score(notes,prediction,clock,profile)
        self.assertEqual(shifted['onset_f1_50ms'],0)


if __name__ == '__main__':
    unittest.main()
