from typing import Protocol, runtime_checkable


@runtime_checkable
class TrainerProtocol(Protocol): ...


@runtime_checkable
class OnPolicyTrainerProtocol(TrainerProtocol, Protocol): ...


@runtime_checkable
class OffPolicyTrainerProtocol(TrainerProtocol, Protocol): ...


@runtime_checkable
class OfflineTrainerProtocol(TrainerProtocol, Protocol): ...
