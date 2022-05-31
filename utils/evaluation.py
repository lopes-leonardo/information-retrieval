import numpy as np

class RetrievalEvaluation:
    
    def __init__(self, ranked_lists:np.ndarray, classes:np.ndarray):
        self.ranked_lists = ranked_lists
        self.classes = classes
        self.class_size = 1000
        self.n = len(ranked_lists)
        
    def p_at_n(self, n:int)->float:
        p_total = 0
        for rank in self.ranked_lists:
            p = 0
            target_class = self.classes[rank[0]]
            for i in range(n):
                if self.classes[rank[i]] == target_class:
                    p += 1
            p = p / n
            p_total += p
        return p_total / self.n
    
    def p_at_10(self)->float:
        return self.p_at_n(n=10)
    
    def p_at_20(self)->float:
        return self.p_at_n(n=20)
    
    def p_at_50(self)->float:
        return self.p_at_n(n=50)
    
    def p_at_100(self)->float:
        return self.p_at_n(n=100)
    
    def computeAveragePrecision(self, rk, d=1000):
        sumrj = 0
        curPrecision = 0
        sumPrecision = 0
        qClass = self.classes[rk[0]]
        for i in range(d):
            imgi = rk[i]
            imgiClass = self.classes[imgi]
            if (qClass == imgiClass):
                sumrj = sumrj + 1
                posi = i + 1
                curPrecision = sumrj / posi
                sumPrecision += curPrecision
        nRel = self.class_size
        l = len(rk)
        avgPrecision = sumPrecision / min(l, nRel)
        return avgPrecision

    def compute_map(self):
        acumAP = 0
        for rk in self.ranked_lists:
            acumAP += self.computeAveragePrecision(rk)
        return acumAP / self.n
    
    def evaluate_all(self) -> None:
        print("=========== Evaluation Procedure ===========")
        print("Evaluation dataset size:", self.n)
        print("Precision at 10:", self.p_at_10())
        print("Precision at 20:", self.p_at_20())
        print("Precision at 50:", self.p_at_50())
        print("Precision at 100:", self.p_at_100())
        # print("Map:", self.compute_map())