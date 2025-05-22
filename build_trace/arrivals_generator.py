import random
import math
import json

from abc import ABC, abstractmethod
from typing import List, Dict, Tuple, Optional, Callable, Any
from pathlib import Path


class ArrivalPattern(ABC):
    """
    Abstract base class for arrival patterns. Each subclass is responsible for
    implementing a method to generate arrival times for a specified number of requests.
    """

    @abstractmethod
    def generate_arrival_times(self, num_requests: int) -> List[float|int]:
        """
        Must return a list of arrival times (floats or ints) for `num_requests` requests.
        """
        raise NotImplementedError

# -------------------------------------------------------
# Continuous Uniform
# -------------------------------------------------------

class ContinuousUniformArrival(ArrivalPattern):
    """
    Requests arrive uniformly in continuous time, within [0, total_time].
    """

    def __init__(self, total_time: float):
        """
        total_time: The time window in which all requests will arrive [0, total_time].
        """
        self.total_time = total_time

    def generate_arrival_times(self, num_requests: int) -> List[float]:
        """
        Generate num_requests times uniformly in [0, total_time].
        """
        return [random.uniform(0, self.total_time) for _ in range(num_requests)]

    def __str__(self):
        return f"ContinuousUniformArrival(total_time={self.total_time})"


# -------------------------------------------------------
# Discrete Uniform
# -------------------------------------------------------

class DiscreteUniformArrival(ArrivalPattern):
    """
    Requests arrive at discrete time steps, each chosen uniformly in [0, max_step].
    So arrival times are integers in [0, max_step].
    """

    def __init__(self, max_step: int):
        """
        max_step: The largest integer time step at which a request can arrive.
        """
        self.max_step = max_step

    def generate_arrival_times(self, num_requests: int) -> List[int]:
        """
        Generate num_requests integer times in [0, max_step].
        """
        return [random.randint(0, self.max_step) for _ in range(num_requests)]

    def __str__(self):
        return f"DiscreteUniformArrival(max_step={self.max_step})"


# -------------------------------------------------------
# Continuous Periodic
# -------------------------------------------------------

class ContinuousPeriodicArrival(ArrivalPattern):
    """
    Continuous periodic: each arrival is exactly interval apart.
    E.g., if interval=2.0, arrivals at times 0, 2, 4, 6, ...
    """

    def __init__(self, interval: float):
        self.interval = interval

    def generate_arrival_times(self, num_requests: int) -> List[float]:
        return [i * self.interval for i in range(num_requests)]

    def __str__(self):
        return f"ContinuousPeriodicArrival(interval={self.interval})"


# -------------------------------------------------------
# Discrete Periodic
# -------------------------------------------------------

class DiscretePeriodicArrival(ArrivalPattern):
    """
    Discrete periodic: arrivals at integer multiples of an interval, but
    we store them as integers. For example, if interval=2, the times are
    [0,2,4,6], etc., all stored as int.
    """

    def __init__(self, interval: int):
        """
        interval: The fixed integer step between consecutive arrivals.
        """
        self.interval = interval

    def generate_arrival_times(self, num_requests: int) -> List[int]:
        return [i * self.interval for i in range(num_requests)]

    def __str__(self):
        return f"DiscretePeriodicArrival(interval={self.interval})"


# -------------------------------------------------------
# Continuous Bimodal
# -------------------------------------------------------

class ContinuousBimodalArrival(ArrivalPattern):
    """
    A bimodal continuous-time arrival process: each inter-arrival time is drawn
    from Exp(rate1) with probability p, or Exp(rate2) with probability (1-p).
    """

    def __init__(self, rate1: float, rate2: float, p: float):
        self.rate1 = rate1
        self.rate2 = rate2
        self.p = p

    def generate_arrival_times(self, num_requests: int) -> List[float]:
        arrivals = []
        current_time = 0.0
        for _ in range(num_requests):
            if random.random() < self.p:
                interarrival = random.expovariate(self.rate1)
            else:
                interarrival = random.expovariate(self.rate2)
            current_time += interarrival
            arrivals.append(current_time)
        return arrivals

    def __str__(self):
        return (f"ContinuousBimodalArrival("
                f"rate1={self.rate1}, rate2={self.rate2}, p={self.p})")


# -------------------------------------------------------
# Discrete Bimodal
# -------------------------------------------------------

class DiscreteBimodalArrival(ArrivalPattern):
    """
    A discrete-time bimodal process: at each integer step t:
     - with probability p, we draw K ~ Poisson(lambda1)
     - with probability (1-p), we draw K ~ Poisson(lambda2)
    Then we place K arrivals at time t.
    """

    def __init__(self, lambda1: float, lambda2: float, p: float, max_steps: int = 100000):
        """
        lambda1, lambda2: The two Poisson rates
        p: fraction of steps that use lambda1 vs. lambda2
        max_steps: safety limit for the # of steps
        """
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.p = p
        self.max_steps = max_steps

    def generate_arrival_times(self, num_requests: int) -> List[int]:
        arrivals = []
        total_requests = 0
        t = 0
        while total_requests < num_requests and t < self.max_steps:
            # decide which distribution to use this step
            if random.random() < self.p:
                k = self._sample_poisson(self.lambda1)
            else:
                k = self._sample_poisson(self.lambda2)

            for _ in range(k):
                arrivals.append(t)
                total_requests += 1
                if total_requests >= num_requests:
                    break
            t += 1
        return arrivals

    @staticmethod
    def _sample_poisson(lmbd: float) -> int:
        """Knuth's algorithm for Poisson(lmbd)."""
        L = math.exp(-lmbd)
        p = 1.0
        k = 0
        while p > L:
            k += 1
            p *= random.random()
        return k - 1

    def __str__(self):
        return (f"DiscreteBimodalArrival("
                f"lambda1={self.lambda1}, lambda2={self.lambda2}, "
                f"p={self.p}, max_steps={self.max_steps})")


# -------------------------------------------------------
# Poisson Bursty
# -------------------------------------------------------
class PoissonBurstyArrivalPattern(ArrivalPattern):
    """
    Generate requests in bursts:
    - Each burst has size ~ Poisson(lambda_burst)
    - Each burst is separated by gap ~ Poisson(lambda_gap)
    """

    def __init__(self, lambda_burst: float, lambda_gap: float = 0):
        self.lambda_burst = lambda_burst
        self.lambda_gap = lambda_gap

    @staticmethod
    def _sample_poisson(lmbd: float) -> int:
        """Knuth's algorithm for Poisson sampling."""
        L = math.exp(-lmbd)
        p = 1.0
        k = 0
        while p > L:
            k += 1
            p *= random.random()
        return k - 1

    def generate_arrival_times(self, num_requests: int) -> List[int]:
        arrivals = []
        current_time = 0
        total_requests = 0
        t = 0

        while total_requests < num_requests:
            burst_size = min(
                self._sample_poisson(self.lambda_burst),
                num_requests - total_requests
            )

            for i in range(burst_size):
                offset = random.randint(int(self.lambda_burst * 0.5), self.lambda_burst)
                arrivals.append(current_time + offset)

            total_requests += burst_size
            # 다음 burst까지의 간격
            gap = int(random.uniform(self.lambda_gap * 0.8, self.lambda_gap))

            current_time += gap
            t += 1

        return sorted(arrivals)

    def __str__(self):
        return (f"PoissonBurstyArrivalPattern("
                f"lambda_burst={self.lambda_burst}, lambda_gap={self.lambda_gap}")


# -------------------------------------------------------
# Continuous Poisson
# -------------------------------------------------------

class ContinuousPoissonArrival(ArrivalPattern):
    """
    Represents a continuous-time Poisson arrival process, sampling inter-arrival
    times from an exponential distribution with parameter 'rate'.
    """

    def __init__(self, rate: float):
        self.rate = rate

    def generate_arrival_times(self, num_requests: int) -> List[float]:
        arrivals = []
        current_time = 0.0
        for _ in range(num_requests):
            interarrival = random.expovariate(self.rate)
            current_time += interarrival
            arrivals.append(current_time)
        return arrivals

    def __str__(self):
        return f"ContinuousPoissonArrival(rate={self.rate})"


# -------------------------------------------------------
# Discrete Poisson
# -------------------------------------------------------

class DiscretePoissonArrival(ArrivalPattern):
    """
    A discrete-time Poisson model where time is considered in discrete steps
    (0, 1, 2, ...). At each integer step t:
      - We draw how many arrivals occur using Poisson(lambda_per_step).
      - All arrivals happening at step t have arrival_time = t.
    """

    def __init__(self, lambda_per_step: float, max_steps: int = 100000):
        self.lambda_per_step = lambda_per_step
        self.max_steps = max_steps

    def generate_arrival_times(self, num_requests: int) -> List[int]:
        arrivals = []
        total_requests = 0
        t = 0
        while total_requests < num_requests and t < self.max_steps:
            k = self._sample_poisson(self.lambda_per_step)
            for _ in range(k):
                arrivals.append(t)
                total_requests += 1
                if total_requests >= num_requests:
                    break
            t += 1
        return arrivals

    @staticmethod
    def _sample_poisson(lmbd: float) -> int:
        L = math.exp(-lmbd)
        p = 1.0
        k = 0
        while p > L:
            k += 1
            p *= random.random()
        return k - 1

    def __str__(self):
        return (f"DiscretePoissonArrival("
                f"lambda_per_step={self.lambda_per_step}, "
                f"max_steps={self.max_steps})")

def extract_params(pattern: ArrivalPattern) -> dict:
    params = {
        "class": pattern.__class__.__name__
    }
    # 각 클래스별 필요한 파라미터 추출
    if isinstance(pattern, DiscretePoissonArrival):
        params.update({
            "lambda_per_step": pattern.lambda_per_step,
            "max_steps": pattern.max_steps
        })
    elif isinstance(pattern, ContinuousPoissonArrival):
        params.update({
            "rate": pattern.rate
        })
    elif isinstance(pattern, DiscreteUniformArrival):
        params.update({
            "max_step": pattern.max_step
        })
    elif isinstance(pattern, ContinuousUniformArrival):
        params.update({
            "total_time": pattern.total_time
        })
    elif isinstance(pattern, DiscretePeriodicArrival):
        params.update({
            "interval": pattern.interval
        })
    elif isinstance(pattern, ContinuousPeriodicArrival):
        params.update({
            "interval": pattern.interval
        })
    elif isinstance(pattern, DiscreteBimodalArrival):
        params.update({
            "lambda1": pattern.lambda1,
            "lambda2": pattern.lambda2,
            "p": pattern.p,
            "max_steps": pattern.max_steps
        })
    elif isinstance(pattern, ContinuousBimodalArrival):
        params.update({
            "rate1": pattern.rate1,
            "rate2": pattern.rate2,
            "p": pattern.p
        })
    elif isinstance(pattern, PoissonBurstyArrivalPattern):
        params.update({
            "lambda_burst": pattern.lambda_burst,
            "lambda_gap": pattern.lambda_gap
        })
    return params

if __name__ == "__main__":
    patterns = [
        DiscreteUniformArrival(max_step=5000),
        # PoissonBurstyArrivalPattern(lambda_burst=5)
        DiscreteBimodalArrival(lambda1=0.005, lambda2=0.0001, p=0.7),
    ]

    filepath = "/home/sychoy/vllm/trace_pool/static_8k_pressure/arrivals.json"

    # JSON 구조 생성
    json_data = [extract_params(p) for p in patterns]

    # 파일로 저장
    Path(filepath).write_text(json.dumps(json_data, indent=2))