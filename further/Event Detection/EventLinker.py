import spacy
import pandas as pd
from datetime import datetime
import re
from spacy.matcher import Matcher
import json
from typing import List, Dict
import numpy as np
from matplotlib import pyplot as plt
from pandas.core.interchange.dataframe_protocol import DataFrame
from spacy.tokens import Doc, Span
from py_heideltime import heideltime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
from toolz.curried import unique


class EventLinker:
    def __init__(self, similarity_threshold=0.5):
        """
        Initialize the event linker component

        Args:
            similarity_threshold: Threshold for considering events as related
        """
        self.similarity_threshold = similarity_threshold
        self.vectorizer = TfidfVectorizer(
            stop_words='english',
            ngram_range=(1, 3),
            min_df=1
        )
        self.event_groups = []
        self.graph = nx.Graph()

    def compute_similarity(self, event1: Dict, event2: Dict) -> float:
        """
        Compute similarity between two events based on text and temporal info
        """

        # Combine event text with relevant context
        event1_text = f"{event1['event_text']} {' '.join([t['text'] for t in event1['temporal_expressions']])}"
        event2_text = f"{event2['event_text']} {' '.join([t['text'] for t in event2['temporal_expressions']])}"

        # check out numerically similarity

        # Convert to TF-IDF vectors
        try:
            vectors = self.vectorizer.fit_transform([event1_text, event2_text])
            similarity = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]
        except:
            similarity = 0

        return similarity

    def are_temporally_related(self, event1: Dict, event2: Dict) -> bool:
        """
        Check if events are temporally related
        """

        if not event1['temporal_expressions'] or not event2['temporal_expressions']:
            return False

        # Check for temporal overlap or proximity (can be improved)
        for temp1 in event1['temporal_expressions']:
            for temp2 in event2['temporal_expressions']:
                if temp1['type'] == temp2['type'] and temp1['value'] == temp2['value']:
                    return True

        return False

    def build_event_graph(self, events: List[Dict]):
        """
        Build a graph of related events
        """

        # Add all events as nodes
        for i, event in enumerate(events):
            self.graph.add_node(i, **event, label=f"{self.graph.number_of_nodes()}")

        # Add edges between related events
        for i in range(len(events)):
            for j in range(i + 1, len(events)):
                event1 = events[i]
                event2 = events[j]

                # Compute similarity
                similarity = self.compute_similarity(event1, event2)

                # Check temporal relation
                temporal_relation = self.are_temporally_related(event1, event2)

                # Add edge if events are related
                if similarity >= self.similarity_threshold or temporal_relation:
                    self.graph.add_edge(i, j,
                                   similarity=similarity,
                                   temporal_relation=temporal_relation)

    def group_related_events(self, events: List[Dict]) -> List[List[Dict]]:
        """
        Group related events together

        Args:
            events: List of event dictionaries

        Returns:
            List of event groups (each group is a list of related events)
        """

        # Build event graph
        self.build_event_graph(events)

        # Find connected components (event groups)
        components = list(nx.connected_components(self.graph))

        # Convert node indices back to events
        self.event_groups = [[events[i] for i in component] for component in components]
        return self.event_groups


class EventDetector:
    def __init__(self,  similarity_threshold=0.5):
        self.linker = EventLinker(similarity_threshold)
        self.nlp = spacy.load('en_core_web_lg')

        self.event_temporal_patterns = {
            'frequency': r'(every|each)\s+(\w+)',
            'duration_range': r'(\d+)\s*-\s*(\d+)\s*(hour|minute|second|day|week|month|year)s?',
            'recurrence': r'(daily|weekly|monthly|yearly|annually)'
        }

        self.verb_matcher = Matcher(self.nlp.vocab)
        self.action_verb_pattern = [
            [
                {"POS": "AUX", "OP": "?"},
                {"POS": "VERB", "OP": "+"},
                {"POS": "NOUN", "OP": "?"}
            ],
            [
                {"POS": "AUX", "OP": "?"},
                {"POS": "VERB"},
                {"POS": "DET", "OP": "?"},
                {"POS": "NOUN", "OP": "+"}
            ],
            [
                {"POS": "AUX", "OP": "?"},
                {"POS": "VERB"},
                {"POS": "ADP"}
            ],
            [
                {"POS": "AUX", "OP": "?"},
                {"POS": "VERB"},
                {"POS": "ADP"},
                {"POS": "DET", "OP": "?"},
                {"POS": "NOUN", "OP": "+"}
            ]
        ]

        self.verb_matcher.add("verb-phrases", self.action_verb_pattern)

        self.conversation_history = []
        self.event_groups = []

    def longest_unique_sequence(self, verbs):
        unique_verb_phrases = []
        for verb_phrase, _ in verbs:
            # Convert verb phrase to string for comparison
            verb_phrase_str = verb_phrase.text

            # Check if this verb phrase is not a subset of any existing unique verb phrase
            if not any(verb_phrase_str in existing for existing in unique_verb_phrases):
                # Remove any existing shorter verb phrases that are substrings
                unique_verb_phrases = [
                    existing for existing in unique_verb_phrases
                    if existing not in verb_phrase_str
                ]

                # Add the current verb phrase
                unique_verb_phrases.append(verb_phrase_str)

        # Now unique_verb_phrases contains the longest unique verb phrases
        return unique_verb_phrases

    def connect_sequences(self, values):
        if len(values) != 0:
            connected_sequence = [values[0]]
            for value in values[1:]:
                last_value = connected_sequence[-1]
                if last_value[1][1] == value[1][0]:
                    connected_sequence.append(value)
            return [" ".join([s[0].text for s in connected_sequence])]
        else:
            return values

    def connect_longest_supersequence(self, values):
        def find_longest_supersequence(values):
            if len(values) != 0:
                longest_values = [values[0]]
                for value in values[1:]:
                    last_value = longest_values[-1]
                    if (last_value[1][0] == value[1][0]) and (last_value[1][1] < value[1][1]):
                        longest_values.remove(last_value)
                        longest_values.append(value)
                    elif (last_value[1][0] < value[1][0]) and (last_value[1][1] == value[1][1]):
                        pass
                    else:
                        longest_values.append(value)
                return longest_values
            else:
                return []

        print("Start here")
        values = find_longest_supersequence(find_longest_supersequence(values))

        if len(values) != 0:
            range_values = [values[0]]

            for value in values[1:]:
                last_value = range_values[-1]

                if value[1][0] == last_value[1][1]:
                    range_values.remove(last_value)
                    value_var = " ".join([str(last_value[0]), str(value[0])])
                    range_values.append((value_var, (last_value[1][0], value[1][1])))
                else:
                    range_values.append(value)
            return range_values
        else:
            return []

    def _get_verb_phrases(self, input_line):
        print(str(input_line))

        doc = self.nlp(input_line.text)

        verbs = [(doc[match[1]:match[2]], match[1:]) for match in self.verb_matcher(doc)]
        #unique_verbs = self.find_longest_supersequence(verbs)
        unique_verbs = self.connect_longest_supersequence(verbs)
        #unique_verbs = self.longest_unique_sequence(verbs)

        print('unique_verbs')
        print(unique_verbs)
        return unique_verbs

    def heideltime_parser(self, text):
        return heideltime(
            text,
            language='english',
            document_type='narrative',
            dct=str(datetime.now()).split(' ')[0]
        )

    def _extract_additional_temporal_patterns(self, text: str) -> Dict:
        """
        Extract additional temporal patterns specific to events.
        """

        temporal_info = {
            'frequency': None,
            'duration_range': None,
            'recurrence': None
        }

        for pattern_type, pattern in self.event_temporal_patterns.items():
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                temporal_info[pattern_type] = matches[0]

        return temporal_info

    def extract_events(self, text: str) -> List[Dict]:
        """
        Extract events and their temporal information from text.

        Args:
            text: Input text to process

        Returns:
            List of dictionaries containing event information
        """
        doc = self.nlp(text)
        events = []

        # Get temporal expressions using selected tagger
        temporal_expressions = self.heideltime_parser(text)

        # Process each sentence
        for sent in doc.sents:
            # Find main verbs and their arguments
            verb_phrases = self._get_verb_phrases(sent)

            # Find temporal expressions within sentence bounds
            sent_temporals = [
                temp for temp in temporal_expressions
                if temp['span'][0] >= sent.start_char and temp['span'][1] <= sent.end_char
            ]

            # Combine verb phrases with temporal information
            for verb_phrase in verb_phrases:
                event = {
                    'event_text': verb_phrase,
                    'sentence': sent.text,
                    'start_position': sent.start_char,
                    'end_position': sent.end_char,
                    'temporal_expressions': sent_temporals
                }

                # Add additional temporal pattern matches
                event.update(self._extract_additional_temporal_patterns(sent.text))
                events.append(event)

        return events

    def create_groups(self, texts: List[str]) -> pd.DataFrame:
        """
        Process a list of texts and return structured event data with event linking.
        """

        all_events = []
        # Process each text as part of conversation
        for text in texts:
            # Extract events
            events = self.extract_events(text)

            # Add to conversation history
            self.conversation_history.extend(events)
            all_events.extend(events)

            # Update event groups
            self.event_groups = self.linker.group_related_events(self.conversation_history)

            # Create DataFrame with event groups
        df = pd.DataFrame(all_events)

        # Add group IDs to events
        df['group_id'] = -1
        for i, group in enumerate(self.event_groups):
            for event in group:
                mask = (df['event_text'] == event['event_text']) & \
                       (df['sentence'] == event['sentence'])
                df.loc[mask, 'group_id'] = i

        return df


if __name__ == '__main__':
    conversation = [
        "A:We need to take the accounts system offline to carry out the upgrade . But don't worry , it won't cause too much inconvenience . We're going to do it over the weekend .",
        "B: How long will the system be down for ?",
        "A: We'll be taking everything offline in about two hours ' time . It'll be down for a minimum of twelve hours . If everything goes according to plan , it should be up again by 6 pm on Saturday ."
        "B: That's fine . We've allowed forty-eight hours to be on the safe side ."
        ]

    # Initialize detector
    detector = EventDetector(similarity_threshold=0.5)

    # Process conversation
    results = detector.create_groups(conversation)

    # Display results with group IDs
    print("\nEvents with group IDs:")
    print(results[['event_text', 'sentence', 'temporal_expressions', 'group_id']])

    # Display event groups
    print("\nEvent Groups:")
    for i, group in enumerate(detector.event_groups):
        print(f"\nGroup {i}:")
        for event in group:
            print(f"- Event: {event['event_text']}")
            print(f"  Context: {event['sentence']}")
            print(f"  Temporal: {[t['text'] for t in event['temporal_expressions']]}")

    G = detector.linker.graph
    pos = nx.spring_layout(G)  # Layout for positioning nodes
    node_labels = nx.get_node_attributes(G, 'label')  # Extract node labels
    nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=300)
    nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=10)

    plt.show()
