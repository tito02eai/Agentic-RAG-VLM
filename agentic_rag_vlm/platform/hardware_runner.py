"""Official Runner UI for a configured site; read-only unless explicitly enabled."""
import json
from pathlib import Path
from urllib.parse import urlsplit

from .config import load_platform_config
from .local_runner import import_official_runner
from .runner_provider import load_provider
from .deployment import validate_profile_fields, validate_profile


class ReadOnlySessionAPI:
    """Default monitor cannot create sessions even through an accidental UI path."""
    supports_trajectories=False
    def __init__(self, api):self._api=api
    def __getattr__(self,name):
        if name.startswith('get_') or name in ('joint_counts','_base_url','set_contact_listener'):
            return getattr(self._api,name)
        if name.startswith(('create_','submit_','stop_')) or name=='open_events':
            raise RuntimeError('Read-only hardware monitor: session creation and commands are disabled')
        raise AttributeError(name)


def validate_hardware_provider(provider):
    profile=provider.profile
    if not profile or profile.get('execution_environment','hardware')!='hardware':
        raise ValueError('a measured hardware profile is required; simulator profiles are not admitted')
    if profile.get('top_rgb_source') not in ('jpeg','paired_rgbd'):
        raise ValueError('confirm the site RGB stream and matching intrinsics before hardware Run')
    validate_profile_fields(profile,robot_id=provider.platform_config.robot_id,arm=profile.get('arm'))
    for key in ('max_joint_delta_deg_per_step','lift_height_m','observation_park_xyz_m','retention_observation'):
        if profile.get(key) is None:raise ValueError('site confirmation pending: '+key)
    if provider.candidate_provider is None:
        raise ValueError('configure and review site RGB candidate regions before first hardware Run')
    if provider.task is None:raise ValueError('first hardware trial requires a fixed measured task')
    if not provider.destinations:raise ValueError('measured destinations required')
    for subgoal in provider.task.subgoals:
        if subgoal.destination not in provider.destinations:raise ValueError('missing task destination')


def make_hardware_playground(checkout,config,output,*,port=18797,enable_run=False):
    config=Path(config).resolve();raw=json.loads(config.read_text())
    platform_path=Path(raw['platform_config'])
    if not platform_path.is_absolute():platform_path=config.parent/platform_path
    platform=load_platform_config(platform_path)
    endpoint=(platform.endpoint_base or '').rstrip('/')
    parsed=urlsplit(endpoint)
    if parsed.scheme not in ('https','http') or not parsed.hostname or parsed.username or parsed.password:
        raise ValueError('valid Session API origin required')
    if parsed.scheme=='http' and parsed.hostname not in ('localhost','127.0.0.1','::1'):
        raise ValueError('remote hardware Session API requires HTTPS')
    _,sessions=import_official_runner(checkout)
    from playground import HostedRunner,RequestError
    from .blupe_client import BluPeSessionClient
    provider=None
    if enable_run:
        provider=load_provider(config);validate_hardware_provider(provider);provider.prepare()
        # Only a public calibration GET here. Home may happen after admission.
        calibration=BluPeSessionClient(endpoint,max_retries=0).get_calibration(platform.robot_id)
        validate_profile(provider.profile,calibration,robot_id=platform.robot_id,arm=provider.profile['arm'])
    origin=f'http://127.0.0.1:{port}'
    def api_factory():
        api=sessions.HttpSessionAPI(endpoint,supports_trajectories=True,robot_id=platform.robot_id,joint_counts=(6,6))
        return api if enable_run else ReadOnlySessionAPI(api)
    def factory(model,directory):
        if not enable_run:raise RuntimeError('hardware Run is disabled')
        value=load_provider(config);validate_hardware_provider(value)
        value.output=Path(directory)/'agentic_runs'
        return value
    class HardwarePlayground(HostedRunner):
        def validate_launch(self,visitor,payload,paid):
            if not enable_run:raise RequestError(409,'Read-only monitor. No queue entry or hardware command is enabled.')
            if payload.get('provider')!='agentic-rag-vlm':raise RequestError(400,'Use the configured Agentic RAG-VLM provider')
            if payload.get('prompt','').strip()!=provider.task.instruction.strip():raise RequestError(400,'Use the fixed measured first-trial task')
            if payload.get('share_conversation') is not False:raise RequestError(400,'First hardware trial keeps model text sharing disabled')
        async def session_extension(self,cookies,visitor):
            return {'local_runner':True,'default_provider':'agentic-rag-vlm','configured_providers':{
                'agentic-rag-vlm':{'label':'Agentic RAG-VLM — hardware '+('Run enabled' if enable_run else 'READ ONLY'),
                    'model':'Agentic RAG-VLM','task':provider.task.instruction if provider else 'Site setup pending — read-only monitor',
                    'help':'Real BluPe camera/queue monitor. Run is disabled.' if not enable_run else 'Real hardware Run joins the public station queue.',
                    'default_run_duration_s':300}}},[]
        def page(self):
            page=super().page()
            banner=('<div style="padding:14px;background:#e5f0f5;color:#123;text-align:center">'
                'Agentic RAG-VLM / '+('REAL HARDWARE RUN ENABLED' if enable_run else 'READ-ONLY HARDWARE MONITOR — Run blocked')+
                ' / '+platform.robot_id+'<br>Real remote cameras; no simulation. '+
                ('Run requests a robot session.' if enable_run else 'Opening this page does not join a queue or send motion.')+'</div>')
            return page.replace(b'<body class="playgroundConsole">',b'<body class="playgroundConsole">'+banner.encode(),1)
    app=HardwarePlayground(public_origin=origin,session_api=endpoint,camera_origin=endpoint,
        development=True,hardware_control=enable_run,robot_id=platform.robot_id,
        api_factory=api_factory,share_conversation=False,recording_root=Path(output)/'recordings',
        extra_providers={'agentic-rag-vlm':factory},camera_names=('top','left','right'),
        robots=[{'id':platform.robot_id,'name':'YAM — '+('HARDWARE' if enable_run else 'READ ONLY'),'url':origin+'/'}])
    return app
