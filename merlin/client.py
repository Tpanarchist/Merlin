from merlin.api.models import ModelsMixin
from merlin.api.responses.responses import ResponsesMixin
from merlin.api.platform.videos import VideosMixin
from merlin.api.platform.images import ImagesMixin
from merlin.api.platform.embeddings import EmbeddingsMixin
from merlin.api.platform.evals import EvalsMixin
from merlin.api.graders import GradersMixin
from merlin.api.fine_tuning import FineTuningMixin
from merlin.api.batch import BatchMixin
from merlin.api.files import FilesMixin
from merlin.api.uploads import UploadsMixin
from merlin.api.moderations import ModerationsMixin
from merlin.api.vector_stores import VectorStoresMixin
from merlin.api.chatkit import ChatKitMixin
from merlin.api.containers import ContainersMixin
from merlin.http_client import MerlinHTTPClient

class MerlinClient(
    ModelsMixin,
    ResponsesMixin,
    VideosMixin,
    ImagesMixin,
    EmbeddingsMixin,
    EvalsMixin,
    GradersMixin,
    FineTuningMixin,
    BatchMixin,
    FilesMixin,
    UploadsMixin,
    ModerationsMixin,
    VectorStoresMixin,
    ChatKitMixin,
    ContainersMixin,
):
    def __init__(self, http: MerlinHTTPClient):
        self._http = http
