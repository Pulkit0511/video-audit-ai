import uuid
import json
import logging
from pprint import pprint

from dotenv import load_dotenv
load_dotenv(override=True)

from backend.src.graph.workflow import app

logging.basicConfig(
    level= logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('video-audit-runner')

def run_cli_simulation():
    '''
    Simulates the video compliance audit request
    '''

    session_id = str(uuid.uuid4())
    logger.info(f'Starting Audit Session: {session_id}')

    initial_inputs = {
        'video_url': 'https://youtu.be/dT7S75eYhcQ',
        'video_id': f'vid_{session_id[:8]}',
        'compliance_results': [],
        'errors': []
    }

    print('\n-------Initializing workflow.......')
    print(f'Input Payload: {json.dumps(initial_inputs, indent=2)}')

    try:
        final_state = app.invoke(initial_inputs)

        audit_result = final_state.get("audit_result")

        if not audit_result:
            print("\nNo audit result produced.")
            return

        print('\n Compliance Audit Report ==')
        print(f'Video ID: {final_state.get("video_id")}')
        print(f'Status: {audit_result.status}')

        print('\n [ VIOLATIONS DETECTED ]')

        results = audit_result.compliance_results

        if results:
            for issue in results:
                print(
                    f'- [{issue.severity}] '
                    f'{issue.category} : {issue.description}'
                )
        else:
            print('No violations found.')

        print('\n[FINAL SUMMARY]')
        print(audit_result.final_report)

    except Exception as e:
        logger.error(f'Workflow Execution Failed: {str(e)}')
        raise e
    
if __name__ == '__main__':
    run_cli_simulation()