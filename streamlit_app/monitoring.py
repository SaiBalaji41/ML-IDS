"""Reusable monitoring table matching packet provenance and actual predictions."""
from datetime import datetime
import html
import json
import pandas as pd
import streamlit as st
from packet_ids.events import decorate_event


def display_time(value):
    try:
        return datetime.fromtimestamp(float(value)).strftime('%H:%M:%S')
    except (TypeError, ValueError, OverflowError):
        return '—'


def event_rows(events):
    rows = []
    for event in reversed(events):
        event = decorate_event(event)
        confidence = event.get('confidence')
        rows.append({'Time': display_time(event.get('timestamp')), 'Source IP': event.get('source_ip') or '—',
            'Protocol': event.get('protocol') or '—', 'True classification': event.get('actual_label') or 'Unknown',
            'Primary model prediction': event.get('predicted_label') or 'Not classified',
            'Confidence': f'{confidence:.2%}' if confidence is not None else '—',
            'Severity': event['severity'], 'Status': event['status'],
            'Destination IP': event.get('destination_ip') or '—',
            'Source': event.get('source', 'Unknown'), 'Sample ID': event.get('sample_id', '—')})
    return rows


def render_feed(events, key, set_payload):
    filters = st.columns([1, 1, 1.4])
    category = filters[0].selectbox('Traffic', ['All traffic', 'Attack predictions', 'Benign predictions', 'Unclassified'], key=f'{key}_traffic')
    status = filters[1].selectbox('Result status', ['All statuses', 'MATCH', 'MISMATCH', 'UNVERIFIED', 'UNCLASSIFIED'], key=f'{key}_status')
    search = filters[2].text_input('Search address or class', key=f'{key}_search').strip().lower()
    selected = []
    for original in events:
        event = decorate_event(original)
        if category == 'Attack predictions' and event.get('is_attack') is not True:
            continue
        if category == 'Benign predictions' and event.get('is_attack') is not False:
            continue
        if category == 'Unclassified' and event.get('predicted_label') is not None:
            continue
        if status != 'All statuses' and event['status'] != status:
            continue
        haystack = ' '.join(str(event.get(k) or '') for k in ('source_ip', 'destination_ip', 'actual_label', 'predicted_label', 'protocol'))
        if search and search not in haystack.lower():
            continue
        selected.append(event)
    columns = ['Time', 'Source IP', 'Protocol', 'True classification', 'Primary model prediction', 'Confidence', 'Severity', 'Status']
    rows = event_rows(selected)
    table = '<div class="event-scroll"><table class="event-table"><thead><tr>'
    table += ''.join(f'<th>{html.escape(c)}</th>' for c in columns) + '</tr></thead><tbody>'
    for row in rows:
        table += '<tr>'
        for column in columns:
            value = html.escape(str(row[column]))
            if column in ('Severity', 'Status'):
                css = {'Critical': 'critical', 'High': 'high', 'Medium': 'medium', 'Review': 'medium',
                       'Info': 'info', 'MATCH': 'info', 'MISMATCH': 'critical'}.get(row[column], 'muted')
                value = f'<span class="event-badge {css}">{value}</span>'
            table += f'<td>{value}</td>'
        table += '</tr>'
    if not rows:
        table += '<tr><td colspan="8" class="event-empty">No packet records match this view.</td></tr>'
    st.html(table + '</tbody></table></div>')
    st.caption(f'{len(rows)} shown · {len(events)} records in this feed. MATCH compares a prediction with a known label; '
               'live and unlabeled captures are UNVERIFIED. Severity is a triage policy, not a calibrated risk score.')
    if rows:
        left, right = st.columns(2)
        left.download_button('Export filtered CSV', pd.DataFrame(rows).to_csv(index=False),
                             f'{key}-events.csv', 'text/csv', key=f'{key}_csv')
        right.download_button('Export event JSON', json.dumps(selected, indent=2),
                              f'{key}-events.json', 'application/json', key=f'{key}_json')
        choices = [i for i, e in enumerate(selected) if e.get('payload_hex')]
        if choices:
            choice = st.selectbox('Inspect a feed packet', choices, key=f'{key}_packet',
                format_func=lambda i: f"{selected[i].get('sample_id')} · {selected[i].get('source_ip') or 'No address'} · {selected[i].get('predicted_label') or 'Unclassified'}")
            if st.button('Use selected packet', key=f'{key}_use'):
                event = selected[choice]
                set_payload(bytes.fromhex(event['payload_hex']), f"{event.get('sample_id')} · {event.get('source')}")
                st.success('Payload selected. Open Packet inspector or Byte explanations in the sidebar.')
        with st.expander('Packet provenance and processing details'):
            st.dataframe(pd.DataFrame([{k: e.get(k) for k in ('sample_id', 'captured_at', 'source_ip', 'destination_ip',
                'source_port', 'destination_port', 'protocol', 'source', 'reason', 'model_sha256')} for e in selected]),
                hide_index=True, width='stretch')
