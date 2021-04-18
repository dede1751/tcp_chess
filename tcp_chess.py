import argparse

from board_gui import ChessGame
import networking as net


def main() -> None:

    parser = argparse.ArgumentParser(
        description="Chess gui with basic multiplayer"
        )
    parser.add_argument(
        "-s", "--server", action="store_true", help="Initialize TCP server."
        )
    parser.add_argument(
        "-p", "--port", action="store", type=int,
        help="Choose portforwarding port (must already be forwarded)."
        )
    parser.add_argument(
        "-r", "--router", action="store", type=str,
        help="Choose target router ip for portforwarding. UPnP must be enabled!."
        )
    parser.add_argument(
        "-c", "--client", action="store", type=str,
        help="Initialize TCP client. Requires server IP."
        )
    args = parser.parse_args()

    if args.server:
        socket = net.server_setup(port=args.port, router=args.router)
    elif args.client:
        socket = net.client_setup(port=args.port, ip=args.client)

    game = ChessGame()

    while True:
        if game.active: # start pygame window
            game.display_frame()
        else:
            if args.server:
                net.server_start(socket, game)
            elif args.client:
                net.client_start(socket, game)


if __name__ == "__main__":
    main()
