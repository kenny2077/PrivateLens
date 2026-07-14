"""Face clustering using DBSCAN on face embeddings."""

import json
import numpy as np
from sqlalchemy.orm import Session

from privatelens.db.schema import get_engine, Face, Person


class FaceClusterer:
    """Cluster face embeddings into people using DBSCAN."""

    def __init__(self, eps: float = 0.5, min_samples: int = 2):
        self.eps = eps
        self.min_samples = min_samples

    def cluster_all(self) -> int:
        """Cluster all unclustered faces into people.

        Returns:
            Number of new people created
        """
        engine = get_engine()
        with Session(engine) as session:
            # Get all faces with embeddings but no cluster
            faces = (
                session.query(Face)
                .filter(Face.embedding.isnot(None), Face.cluster_id.is_(None))
                .all()
            )

            if len(faces) < self.min_samples:
                return 0

            try:
                return self._sklearn_cluster(session, faces)
            except ImportError:
                return self._simple_cluster(session, faces)

    def _simple_cluster(self, session: Session, faces: list[Face]) -> int:
        """Fallback clustering using cosine similarity threshold."""
        # Decode embeddings
        embedding_list: list[np.ndarray] = []
        valid_faces: list[Face] = []
        for face in faces:
            try:
                if face.embedding is None:
                    continue
                emb = np.frombuffer(face.embedding, dtype=np.float32)
                if emb.shape[0] == 512:
                    embedding_list.append(emb)
                    valid_faces.append(face)
            except Exception:
                continue

        if len(embedding_list) < self.min_samples:
            return 0

        embeddings = np.array(embedding_list)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1
        embeddings = embeddings / norms

        # Greedy clustering
        people_created = 0
        assigned = [False] * len(valid_faces)

        for i in range(len(valid_faces)):
            if assigned[i]:
                continue

            # Find all similar faces
            cluster_indices = [i]
            assigned[i] = True

            for j in range(i + 1, len(valid_faces)):
                if assigned[j]:
                    continue
                sim = np.dot(embeddings[i], embeddings[j])
                if sim >= (1 - self.eps):
                    cluster_indices.append(j)
                    assigned[j] = True

            if len(cluster_indices) >= self.min_samples:
                person = Person(
                    display_name=f"Person {people_created + 1}",
                    user_labeled=False,
                )
                session.add(person)
                session.flush()

                for idx in cluster_indices:
                    valid_faces[idx].cluster_id = person.id

                person.face_count = len(cluster_indices)
                people_created += 1

        session.commit()
        return people_created

    def _sklearn_cluster(self, session: Session, faces: list[Face]) -> int:
        """Cluster using DBSCAN from sklearn."""
        from sklearn.cluster import DBSCAN

        # Decode embeddings
        embedding_list: list[np.ndarray] = []
        valid_faces: list[Face] = []
        for face in faces:
            try:
                if face.embedding is None:
                    continue
                emb = np.frombuffer(face.embedding, dtype=np.float32)
                if emb.shape[0] == 512:
                    embedding_list.append(emb)
                    valid_faces.append(face)
            except Exception:
                continue

        if len(embedding_list) < self.min_samples:
            return 0

        embeddings = np.array(embedding_list)

        # Normalize
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1
        embeddings = embeddings / norms

        # Cluster
        clustering = DBSCAN(eps=self.eps, min_samples=self.min_samples, metric="cosine").fit(
            embeddings
        )
        labels = clustering.labels_

        # Create people for each cluster
        people_created = 0
        for label in set(labels):
            if label == -1:
                continue  # Noise

            person = Person(
                display_name=f"Person {label + 1}",
                user_labeled=False,
            )
            session.add(person)
            session.flush()

            # Assign faces to this person
            for i, face in enumerate(valid_faces):
                if labels[i] == label:
                    face.cluster_id = person.id

            # Update face count
            person.face_count = session.query(Face).filter_by(cluster_id=person.id).count()
            people_created += 1

        session.commit()
        return people_created

    def assign_name(self, person_id: int, name: str) -> bool:
        """Assign a user-provided name to a person cluster."""
        engine = get_engine()
        with Session(engine) as session:
            person = session.query(Person).filter_by(id=person_id).first()
            if not person:
                return False
            person.display_name = name
            person.user_labeled = True
            session.commit()
            return True
