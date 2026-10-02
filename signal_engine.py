import asyncio
import hashlib
import logging
import math
import random
import time
from dataclasses import dataclass
from functools import lru_cache


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger("SignalEngine")


@dataclass(frozen=True)
class Signal:
    identifier: str
    timestamp: float
    values: tuple[float, ...]


@dataclass
class Result:
    identifier: str
    score: float
    confidence: float
    entropy: float
    fingerprint: str
    latency: float


@lru_cache(maxsize=4096)
def normalise(values: tuple[float, ...]) -> tuple[float, ...]:
    magnitude = math.sqrt(
        sum(value ** 2 for value in values)
    ) or 1.0

    return tuple(
        value / magnitude
        for value in values
    )


def entropy(values: tuple[float, ...]) -> float:
    total = sum(abs(x) for x in values) or 1.0

    probabilities = [
        abs(x) / total
        for x in values
        if x != 0
    ]

    return -sum(
        p * math.log2(p)
        for p in probabilities
    )


def fingerprint(signal: Signal) -> str:
    data = (
        f"{signal.identifier}:"
        f"{signal.timestamp}:"
        f"{signal.values}"
    )

    return hashlib.sha256(
        data.encode()
    ).hexdigest()


class AdaptiveModel:

    def __init__(self):
        self.state = random.uniform(-1, 1)
        self.decay = 0.94
        self.temperature = 0.72

    def activate(self, value: float) -> float:
        return 1 / (
            1 + math.exp(
                -value / self.temperature
            )
        )

    def predict(
        self,
        values: tuple[float, ...]
    ) -> float:

        values = normalise(values)

        transformed = [
            math.tanh(
                math.sin(x * math.pi)
                + math.log1p(abs(x))
            )
            for x in values
        ]

        weighted = sum(
            value * self.decay ** index
            for index, value
            in enumerate(transformed)
        )

        self.state = (
            self.decay * self.state
            + (1 - self.decay) * weighted
        )

        return self.activate(
            weighted * (1 + abs(self.state))
        )


async def generate_signals(count: int):

    for index in range(count):

        await asyncio.sleep(
            random.uniform(0.001, 0.005)
        )

        yield Signal(
            identifier=f"SIG-{index:06d}",
            timestamp=time.time(),
            values=tuple(
                random.gauss(0, 1)
                for _ in range(32)
            )
        )


class ProcessingEngine:

    def __init__(self):
        self.model = AdaptiveModel()

    async def process(
        self,
        signal: Signal
    ) -> Result:

        start = time.perf_counter()

        score = self.model.predict(
            signal.values
        )

        signal_entropy = entropy(
            signal.values
        )

        confidence = min(
            0.999,
            0.5
            + abs(score - 0.5)
            + signal_entropy / 25
        )

        elapsed = (
            time.perf_counter() - start
        ) * 1000

        return Result(
            identifier=signal.identifier,
            score=round(score, 6),
            confidence=round(confidence, 6),
            entropy=round(
                signal_entropy,
                6
            ),
            fingerprint=fingerprint(signal),
            latency=round(
                elapsed,
                4
            )
        )


async def main():

    logger.info(
        "Initialising adaptive signal engine..."
    )

    engine = ProcessingEngine()
    results = []

    start = time.perf_counter()

    async for signal in generate_signals(1000):

        result = await engine.process(
            signal
        )

        results.append(result)

        if len(results) % 100 == 0:
            logger.info(
                "Processed %d signals",
                len(results)
            )

    elapsed = (
        time.perf_counter() - start
    )

    scores = [
        result.score
        for result in results
    ]

    logger.info(
        "========================================"
    )

    logger.info(
        "PROCESSING COMPLETE"
    )

    logger.info(
        "Signals processed : %d",
        len(results)
    )

    logger.info(
        "Mean score        : %.6f",
        sum(scores) / len(scores)
    )

    logger.info(
        "Maximum score     : %.6f",
        max(scores)
    )

    logger.info(
        "Minimum score     : %.6f",
        min(scores)
    )

    logger.info(
        "Execution time    : %.4fs",
        elapsed
    )

    logger.info(
        "Throughput        : %.2f signals/sec",
        len(results) / elapsed
    )

    logger.info(
        "========================================"
    )


if __name__ == "__main__":
    asyncio.run(main())