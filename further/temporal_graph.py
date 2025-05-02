import numpy as np

class Interval:
    def __init__(self, low, high):
        self.low, self.high = low, high

    def __eq__(self, interval):
        return (self.low == interval.low) and (self.high == interval.high)

    def contains(self, interval):
        return (self.low <= interval.low) and (self.high >= interval.high)

    def present(self, interval):
        return (self.low > interval.low) and (self.high < interval.high)

    def overlap(self, interval):
        return (interval.low <= self.low <= interval.high) or (interval.low <= self.high <= interval.high)

    def diff(self):
        return self.high - self.low

class GraphNode:
    def __init__(self, event: str):
        if not isinstance(event, str):
            raise Exception("event has to be string type")

        self.event = event
        self.temporal_range = []    # stores temporal values like points and ranges
        self.phrase = []            # list of phrase strings
        self.embedding = []         # list of raw embedding vectors
        self.avg_embedding = None   # running centroid of all embeddings
        self.metadata = []

    def add_temporal_point(self, temporal_point: int):
        """
            Adds temporal point to the node
            :param: specific time point
        """
        
        if not isinstance(temporal_point, int):
            raise Exception("temporal_point has to be int type")
        self.temporal_range.append(temporal_point)

    def add_phrase(self, phrase: str, embedding):
        """
            Adds a verb phrase into the node
            :param phrase: phrase to be added
            :param embedding: embedding of the phrase
        """
        
        # store phrase text
        if not isinstance(phrase, str):
            raise Exception("phrase has to be str type")
        self.phrase.append(phrase)

        # store raw embedding
        self.embedding.append(embedding)

        # recompute running average centroid
        stacked = np.stack(self.embedding, axis=0)
        self.avg_embedding = np.mean(stacked, axis=0)

    def add_metadata(self, metadata):
        self.metadata.append(metadata)

    def __str__(self):
        return f"{self.event}"

class TemporalGraph:
    def __init__(self, sim=None, thres=0.5):
        self.__events = dict()
        self._sim = sim
        self._thres = thres

    def add_event(self, event: str, temporal_point=None, phrase=None, metadata=None):
        """
            Adds a new event in the graph
        """
        if not self.check_event(event):  # if event doesn't exist in the graph
            eventNode = GraphNode(event)
            self.__events[eventNode] = []  # add the event in the adjacency list

        if temporal_point is not None:  # this event has been observed with a different temporal_point
              self.update_time(event, temporal_point)

        if phrase is not None:  # this event has been observed with a different phrase
            self.update_phrase(event, phrase)

        if metadata is not None:  # this event has been observed with a different metadata
            self.update_metadata(event, metadata)

    def add_dependence(self, event1: str, event2: str):
        """
            Adds an edge between two existing nodes
        """
        if not self.check_event(event1):  # check if event1 exists
          raise Exception("Event1 doesn't exist")
        if not self.check_event(event2):  # check if event2 exists
            raise Exception("Event2 doesn't exist")

        eventNode1, eventNode2 = self.get_event(event1), self.get_event(event2)
        self.__events[eventNode1].append(eventNode2)

    def get_event(self, event: str):
        """
            Returns a present event in the graph
        """

        for eventNode in self.__events:
            if str(eventNode) == event:
                return eventNode
        else:
            raise Exception("Node doesn't exist")

    def get_event_nodes(self, event: str):
        """
            Returns all nodes in the graph with given event
        """

        if self.check_event(event):
            return [eventNode for eventNode in self.__events if str(eventNode) == event]
        else:
            raise Exception("Node doesn't exist")

    def check_event(self, event: str):
        """
            Checks if a node of the given event is present or not
        """
        return any([str(eventNode) == event for eventNode in self.__events])

    def update_time(self, event:str, temporal_point):
        """
            Updates temporal values for the event
        """
        for eventNode in self.__events:
            if str(eventNode) == event:
                if isinstance(temporal_point, Interval):
                    eventNode.add_temporal_point(temporal_point.diff())
                else:
                    eventNode.add_temporal_point(temporal_point)
                break
        else:
            raise Exception("Node doesn't exist")

    def update_phrase(self, event: str, phrase: str):
        """
        Add a new phrase to the appropriate GraphNode(event):
         0) If this exact phrase was already seen under any 'event' node → do nothing.
         1) If the node has no phrases yet → just add it.
         2) If embedding(sim, avg_embedding) > threshold → merge into that node.
         3) Otherwise → spawn a new sibling node(event).
        """
        # 0) encode the incoming phrase once
        p_emb = sbert_model.encode(phrase)

        # 0a) dedupe: if ANY existing node for 'event' already has this phrase, skip.
        for node in self.__events:
            if node.event == event and phrase in node.phrase:
                return  # we’ve already stored this exact phrase

        # 1) find your event-node(s) and decide where to put it
        for node in self.__events:
            if node.event == event:
                # 1a) first-ever phrase for this node? just add
                if node.avg_embedding is None:
                    node.add_phrase(phrase, p_emb)

                # 1b) similar enough to this node’s centroid? merge in
                elif self.similiar(p_emb, node.avg_embedding):
                    node.add_phrase(phrase, p_emb)

                # 1c) too different → create a brand-new sibling node(event)
                else:
                    new_node = GraphNode(event)
                    new_node.add_phrase(phrase, p_emb)
                    self.__events[new_node] = []

                return

        # 2) if we never found an 'event' node at all, that’s an error
        raise Exception("Node doesn't exist")


    def update_metadata(self, event: str, metadata):
        """
            Updates metadata for the event
        """
        for eventNode in self.__events:
            if str(eventNode) == event:
                    eventNode.add_metadata(metadata)
                    break
        else:
            raise Exception("Node doesn't exist")

    def similiar(self, emb1, emb2):
        return self._sim(emb1, emb2) > self._thres

    def __str__(self):
        """
            Convert the graph into string
        """
        s = ""
        for i in self.__events:
            s+= f"{i} : {i.temporal_range} : {[str(j) for j in self.__events[i]]} : {i.phrase} : {i.metadata}\n"
        return s

    def serialize(self):
        """
            Serialize the graph
        """
        dic = dict()
        for i in self.__events:
            dic[str(i)] = {'ngb': set(), 'time_point': []}
            for j in self.__events[i]:
                dic[str(i)]['ngb'].add(str(j))
            dic[str(i)]['time_point'] = i.temporal_range
            dic[str(i)]['ngb'] = list(dic[str(i)]['ngb'])
            dic[str(i)]['phrase'] = i.phrase
            dic[str(i)]['metadata'] = i.metadata
        return dic

    def get_next_event(self, event: str):
        """
            Returns the neighbors of the event
            :param event: event for the node
            :return: neighbors of the event as List[GraphNode]
        """
        
        if isinstance(event, str):
            eventNode = self.get_event(event)
        elif isinstance(event, GraphNode):
            eventNode = event
        return self.__events[eventNode]

    def get_all_events(self):
        """
           Returns all nodes in the graph
           :return: List[GraphNode]
        """
        
        return list(self.__events.keys())
    
    def from_serialized(cls, serialized: dict, sim, thres=0.5):
        """
        Rebuilds a TemporalGraph from the output of .serialize().

        Args:
          serialized: dict mapping event_str → {
              'ngb': [neighbor_event_str, ...],
              'time_point': [int, ...],
              'phrase': [str, ...],
              'metadata': [...],
          }
          sim:   similarity func (e.g. cosine_similarity)
          thres: threshold used originally
        """
        # 1) create empty graph
        tg = cls(sim=sim, thres=thres)

        # 2) instantiate one GraphNode per event_str
        for ev in serialized:
            node = GraphNode(ev)
            tg._TemporalGraph__events[node] = []

        # 3) refill each node’s data
        for node in list(tg._TemporalGraph__events):
            info = serialized[node.event]
            node.temporal_range = list(info.get("time_point", []))
            node.metadata       = list(info.get("metadata", []))
            node.phrase         = list(info.get("phrase", []))
            # re-encode embeddings for each phrase
            node.embedding      = [sbert_model.encode(p) for p in node.phrase]
            # recompute centroid
            if node.embedding:
                arr = np.stack(node.embedding, axis=0)
                node.avg_embedding = np.mean(arr, axis=0)
            else:
                node.avg_embedding = None

        # 4) rebuild edges
        for node in list(tg._TemporalGraph__events):
            for nbr_str in serialized[node.event].get("ngb", []):
                nbr_node = next(n for n in tg._TemporalGraph__events
                                if n.event == nbr_str)
                tg._TemporalGraph__events[node].append(nbr_node)

        return tg
