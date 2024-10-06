import math

class Interval:
    """
        For interval operations
    """    

    def __init__(self, low, high):
        self.low, self.high = low, high
        self.value = low, high, self.diff()
    
    def __eq__(self, interval)->bool:
        return (self.low == interval.low) and (self.high == interval.high)
    
    def contains(self, interval)->bool:
        return (self.low <= interval.low) and (self.high >= interval.high)
    
    def present(self, interval)->bool:
        return (self.low > interval.low) and (self.high < interval.high)
    
    def overlap(self, interval)->bool:
        return (interval.low <= self.low <= interval.high) or (interval.low <= self.high <= interval.high)
    
    def diff(self)->float:
        return abs(self.high - self.low)

class GraphNode:
    """
        Each node has an event and a temporal point
        Temporal point is either as a point or an interval
    """

    def __init__(self, event: str):
        if not isinstance(event, str):
            raise Exception("event has to be string type")

        self.event = event
        self.temporal_range = []

    def add_temporal_point(self, temporal_point: int):
        if isinstance(temporal_point, int):
            self.temporal_range.append(temporal_point)
        elif isinstance(temporal_point, Interval):
            self.temporal_range.append(temporal_point.value)
        else:
            raise Exception("temporal_point has to be int or interval type")

    def __str__(self)->str:
        return f"{self.event}"

class TemporalGraph:
    """
        Graph to store the relations between events wrt time
    """

    def __init__(self):
        self.__events = dict()

    def add_event(self, event: str, temporal_point=None)->GraphNode:
        if not self.check_event(event):  # event not present
            eventNode = GraphNode(event.strip())
            self.__events[eventNode] = []
        if temporal_point is not None:  # update event temporal point
              self.update_time(event.strip(), temporal_point)
        return self.get_event(event.strip())

    def add_dependence(self, causal_event: str, effect_event: str)->None:
        if not self.check_event(causal_event.strip()):
            print(causal_event.strip())
            raise Exception("Event1 doesn't exist")
        if not self.check_event(effect_event.strip()):
            raise Exception("Event2 doesn't exist")

        causal_node, effect_node = self.get_event(causal_event.strip()), self.get_event(effect_event.strip())
        self.__events[causal_node].append(effect_node)

    def get_event(self, event: str)->GraphNode:
        for eventNode in self.__events:
            if str(eventNode) == event.strip():
                return eventNode
        else:
            raise Exception("Node doesn't exist")

    def check_event(self, event: str)->bool:
        return any([str(eventNode) == event.strip() for eventNode in self.__events])

    def update_time(self, event:str, temporal_point):
        for eventNode in self.__events:
            if str(eventNode) == event.strip():
                eventNode.add_temporal_point(temporal_point)
                break
        else:
            raise Exception("Node doesn't exist")

    def __str__(self)->str:
        s = ""
        for i in self.__events:
            s += f"{i} : {str(i.temporal_range)}"
            for j in self.__events[i]:
                s+= f"\n\t\t{str(j)}\n"
        return s
    
    def get_complete_graph(self)->dict:
        return self.__events
    
    def get_all_events(self)->list:
        return list(self.__events.keys())
                        