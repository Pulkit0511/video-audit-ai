import operator
from typing import Annotated, List, Dict, Optional, Any, TypedDict
from pydantic import BaseModel, Field

class ComplianceIssue(TypedDict):
    category: str
    description: str
    severity: str
    timestamp: Optional[str]

class VideoAuditState(TypedDict, total=False):
    video_url: str
    video_id: str

    local_file_path: Optional[str]
    video_metadata: Optional[Dict[str, Any]]
    transcript: Optional[str]
    ocr_text: List[str]

    compliance_results: Annotated[List[ComplianceIssue], operator.add]

    # final_status: Optional[str]
    # final_report: Optional[str]
    audit_result: Optional["AuditResult"]

    errors: Annotated[List[str], operator.add]

class ComplianceIssueModel(BaseModel):
    category: str
    description: str
    severity: str
    timestamp: Optional[str] = None

class AuditResult(BaseModel):
    compliance_results: List[ComplianceIssueModel] = Field(default_factory=list)
    status: str
    final_report: str