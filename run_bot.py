from argparse import ArgumentParser

from environment import WizardEnv, RoomGroup
from actor_critic import ActorCriticAgent


def main(url : str, room_id : str, name : str, checkpoint : str, delay : float):
    state_size = WizardEnv.OBS_DIM
    shared_agent = ActorCriticAgent(state_size, False, checkpoint_path=checkpoint)
    shared_agent.load_weights()
    rg = RoomGroup()
    rg.publish(0, room_id)
    env = WizardEnv(name, False, shared_agent, websocket_url=url,
                room_group=rg, create_room=False, close_delay=0, think_delay=delay)

    env.play_game()

if __name__ == "__main__":
    p = ArgumentParser()
    p.add_argument('-u', '--url', required=True)
    p.add_argument('-r', '--room', required=True)
    p.add_argument('-n', '--name', required=True)
    p.add_argument('-c', '--checkpoint', required=True)
    p.add_argument('-d', '--think-delay', type=float, required=True)
    args = p.parse_args()
    main(args.url, args.room, args.name, args.checkpoint, args.think_delay)
