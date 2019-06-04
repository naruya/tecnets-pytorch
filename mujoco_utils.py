from gym.envs.mujoco.pusher import PusherEnv

def load_env(xml_path):
    xml_filepath = 'sim_push_xmls/' + xml_path.split("/")[-1]
    env = PusherEnv(**{'xml_file':xml_filepath, 'distractors': True})
    return env
