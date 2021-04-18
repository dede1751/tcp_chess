# TCP_CHESS
Chess gui with basic legality checking (fully legal move checking is implemented, but very inefficient) that allows for p2p play through a tcp connection. Port forwarding is automated through UPnP (must be enabled on IGD) on port 56969. This project is very rough around the edges and mainly just a quick way for me to play chess with my friends and apply some concepts from our networking course.

## To run:

First run the server script.
  If you have multiple gateway devices on your local network, specify the one with WAN access using the -r argument.
  If you do not wish to use UPnP and manually forwarded the port, simply add the -p tag with the port number:
```
python tcp_chess.py -s -r 192.168.1.1

> Server listening on socket 192.168.1.9:56969 . . . 
```

Using this server address, have the other person run the client script. If portforwarding was done manually, specify the connection port with -p:
```
python tcp_chess.py -c 192.168.1.9
```

## In game:

To quit use Q or the exit window button. To be able to play again, both players must quit the game and then the server will be able to start a new game through standard input.
