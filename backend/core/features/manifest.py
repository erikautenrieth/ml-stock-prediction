import hashlib
import json

from pydantic import BaseModel


class FeatureManifest(BaseModel):
    columns: list[str]
    row_count: int
    column_hash: str

    @classmethod
    def from_dataframe(cls, df) -> "FeatureManifest":
        cols = sorted(df.columns.tolist())
        col_hash = hashlib.sha256(json.dumps(cols).encode()).hexdigest()[:16]
        return cls(columns=cols, row_count=len(df), column_hash=col_hash)

    def validate_against(self, df) -> list[str]:
        current = self.from_dataframe(df)
        errors = []
        if current.column_hash != self.column_hash:
            missing = set(self.columns) - set(current.columns)
            extra = set(current.columns) - set(self.columns)
            if missing:
                errors.append(f"Missing columns: {missing}")
            if extra:
                errors.append(f"Extra columns: {extra}")
        return errors
