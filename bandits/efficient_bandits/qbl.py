from .pqueue import PriorityQueue
import random

#import cProfile

class QBLBandit(object):
    def __init__(self, kg, model_path=None, initial_entities=None, gamma=0.1) -> None:
        self.arms: list[int] = []
        #random.shuffle(self.arms)
        #for i, arm in enumerate(self.arms):
        #    arm.id = i
        self.reward_max = 1
        self.k: int = kg.number_of_triples
        self.kg = kg
        self.weights = []
        self.gamma = gamma 

        if model_path is not None:
            raise Exception("Not implemented") 
        elif initial_entities is None:
            self.arms = list(range(self.k))
        elif initial_entities is not None:
            raise Exception("Not implemented") 
        



        self.in_active_term: dict[int, bool] = {arm: False for arm in self.arms}
        self.last_term_reward: dict[int, float] = {arm: 1.0 for arm in self.arms}
        self.last_term_length: dict[int, int] = {arm: 1 for arm in self.arms}
        self.total_last_term_reward: float = float(self.k)
        self.total_last_term_length: int = self.k
        
        self.queue: PriorityQueue = PriorityQueue(
            [(i, arm) for i, arm in enumerate(self.arms)]
        )
        
        
                        #arm to prio
        self.priority: dict[int, int] = {arm: i for i, arm in enumerate(self.arms)}

    def choose_k(self, m: int, debug = False):
        #selected: list[int] = []
        #for (p,v) in self.queue.lazy_topm(m):
        #    selected.append(v)
        #cnt = self.queue.cnt
        #self.queue.reset_cnt()
        #return selected, cnt
        selected: list[int] = []
        popped: list[tuple[int, int]] = []
        
        for _ in range(m):
            pop: tuple[int, int] = self.queue.pop()  # type: ignore
            popped.append(pop)
            selected.append(pop[1])

        i: tuple[int, int]
        for i in popped:
            self.queue.put(i)  # type: ignore

        #print()
        #print(f"Selected: {selected}")
        #print(f"TOP: {self.queue._heap[0:m]}")
        cnt = self.queue.cnt
        self.queue.reset_cnt()
        return selected, cnt

    def update(
        self, arms_played: list[int], arms_reward: list[float]
    ) -> None:
        
        #logging.info(f"Arm rewards keys: {arms_reward.keys()}")
        #logging.info(f"Arm rewards values: {arms_reward.values()}")
    
        for arm_id in arms_played:
            #TODO: THIS IS COPYPASTE THAT WILL BREAK THE CODE IF WE ACTUALLY SEND MORE NOW
            #arm: int =  self.arms[arm_id]
            idx: int = arm_id
            #logging.info(f'Arm played: {arm_id}, {idx}')
            if not self.in_active_term[idx]:
                self.total_last_term_reward = self.total_last_term_reward-self.last_term_reward[idx]+self.last_term_reward[idx]/self.last_term_length[idx]
                self.last_term_reward[idx] = self.last_term_reward[idx]/self.last_term_length[idx]
                self.total_last_term_length = self.total_last_term_length-self.last_term_length[idx]+1
                self.last_term_length[idx] = 1
                self.in_active_term[idx] = True

             
            self.total_last_term_reward += arms_reward[0] #TODO: FIXED TO SINGLE UPDATE AT THE TIME
            self.last_term_reward[idx] += arms_reward[0]
            self.total_last_term_length += 1
            self.last_term_length[idx] += 1

            weighted_global_avg: float = self.total_last_term_reward / float(
                self.total_last_term_length
            )
            local_avg: float = self.last_term_reward[idx] / float(self.last_term_length[idx])

            is_rewarding: bool = (
                weighted_global_avg
                < local_avg * random.uniform(1-self.gamma,1+self.gamma)   
            )
            
            #logging.info(f'Global avg: {weighted_global_avg}, Local_avg: {local_avg}')
            if not is_rewarding:
                old_prio = self.priority[idx]
                # NOTE: Implement a .top() for the queue to avoid pop put
                top: tuple[int, int] = self.queue._heap[0]  # type: ignore
                #self.queue.put(top)  # type: ignore
                #new_prio: int = self.last_term_length[idx] - 1
                self.priority[idx] = min(self.priority[idx]-1, top[0]-1+self.last_term_length[idx]-self.k)
                #logging.info(f'Not rewarding. Prio before: {old_prio}, New prio: {self.priority[idx]}')
                #self.queue.put((self.priority[idx],top[1])) #type: ignore
                self.queue.update_elem(idx, (self.priority[idx], idx))  # type: ignore
                self.in_active_term[idx] = False
            

    def create_binary_rewards(self, queries, summary):
       
        queries_set = set()
        for q in queries:
            queries_set.add(q)
        queries = queries_set
            
        regrets = []
        #print("\n\n\n")
        for (e1, r, e2) in summary:
            e1 = self.kg.entity_to_id[e1]
            e2 = self.kg.entity_to_id[e2]
            r = self.kg.relationship_to_id[r]
            choice_index = self.kg.triple_to_id[(e1, r, e2)]
            #print(choice_index)
            reward = 1 if (e1 in queries or e2 in queries) else 0
            self.update([choice_index], [reward])
            regrets.append(reward)
        cnt = self.queue.cnt
        self.queue.reset_cnt()
        return regrets, cnt
    
    def create_rewards(self, queries, summary):
        queries_set = set()
        for q in queries:
            queries_set.add(q)
        queries = queries_set

        summary_entities = set()
        for (e1,_,e2) in summary:
            e1 = self.kg.entity_to_id[e1]
            e2 = self.kg.entity_to_id[e2]
            summary_entities.add(e1)
            summary_entities.add(e2)

        regrets = []
        for _ in range(1):
            for e1 in queries:
                if e1 not in self.kg.triples:
                    continue
                for r in self.kg.triples[e1]:
                        for e2 in self.kg.triples[e1][r]:
                            choice_index = self.kg.triple_to_id[(e1, r, e2)]
                            reward = 0

                            #We throw away knowledge we should not have. Aka. 
                            if e1 in queries and e1 in summary_entities:
                                reward += 1/3
                                if e2 in queries and e2 in summary_entities:
                                    reward += 2/3 


                            #We may adjust the bandit with this knowledge as we can deduce it if we know the graph setting
                            if (self.kg.id_to_entity[e1],self.kg.id_to_relationship[r],self.kg.id_to_entity[e2]) in summary:
                                continue
                                #self.update([choice_index], [reward])
                                #regrets.append(1-reward)
                            else: 
                                #self.update([choice_index], [reward])
                                #regrets.append(1-reward)
                                self.total_last_term_reward += reward
                                self.last_term_reward[choice_index] += reward
                                #self.total_last_term_length += 1
                                #self.last_term_length[choice_index] += 1

        for _ in range(1):
            #Normal update of QBL
            for (e1, r, e2) in summary:
                e1 = self.kg.entity_to_id[e1]
                e2 = self.kg.entity_to_id[e2]
                r = self.kg.relationship_to_id[r]

                choice_index = self.kg.triple_to_id[(e1, r, e2)]
                reward = 0
                if e1 in queries:
                    reward += 1/3
                    if e2 in queries:
                        reward += 2/3
                self.update([choice_index], [reward])
                regrets.append(1-reward)
        

                        

        cnt = self.queue.cnt
        self.queue.reset_cnt()
        return regrets, cnt