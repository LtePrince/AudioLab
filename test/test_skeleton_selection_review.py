from script.review_skeleton_selection import difference, clips


def test_identity_uses_dense_frames_not_renumbered_event_ids():
    arrays=dict(event_times_seconds=[.1,.2,.3],valid_event_time=[True]*3)
    baseline=dict(events=[dict(id=f'event-{i}',time_seconds=t,confidence=.9) for i,t in enumerate([.1,.2,.3])])
    candidate=dict(events=[dict(id=f'event-{i}',time_seconds=t,confidence=.9) for i,t in enumerate([.1,.3])])
    rows=difference(arrays,baseline,candidate)
    assert [r['status'] for r in rows]==['retained','deleted','retained']
    assert rows[2]['baseline_event_id']=='event-2' and rows[2]['candidate_event_id']=='event-1'
    assert rows[2]['frame_indices']==[2]


def test_duplicate_time_keeps_all_source_frames():
    arrays=dict(event_times_seconds=[.1,.1],valid_event_time=[True,True])
    selected=dict(events=[dict(id='event-0',time_seconds=.1,confidence=.9)])
    assert difference(arrays,selected,selected)[0]['frame_indices']==[0,1]


def test_first_middle_last_clip_selection_clips_and_merges_overlap():
    changes=[dict(status='deleted',time_seconds=t,source_id=str(t)) for t in [1,4,9,12,19]]
    spans=clips(changes,20)
    assert spans==[dict(start_seconds=0,end_seconds=13,source_ids=['1','9']),
                   dict(start_seconds=15,end_seconds=20,source_ids=['19'])]
