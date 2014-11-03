import redis
from refinery import app
from flask import Response

'''

Handles pubsub stuff

'''

# START REDIS for pubsub system, should only happen once
msgServer = redis.StrictRedis(socket_timeout=20)

#Generic function to call redis and start pub/sub messaging service
def event_stream(channel=None):
    
    pubsub = msgServer.pubsub()
    pubsub.subscribe(channel)
    
    # handle client disconnection in the client side by calling the exit keyword

    try:
        for msg in pubsub.listen():
            yield 'data: %s\n\n' % msg['data']
    except Exception:
        yield 'data: NONE\n\n' #if a timeout happens on the listen, we need to return something


'''

These are the pubsub channels that serve the information

'''
    
@app.route("/<username>/stream_upload")
def stream_upload(username=None):
    mimetype = "text/event-stream"
    channel = username + "Xupload"
    return Response(event_stream(channel=channel), mimetype=mimetype)

@app.route("/<username>/stream_menus")
def stream_menus(username=None):
    mimetype = "text/event-stream"
    channel = username + "Xmenus"
    return Response(event_stream(channel=channel), mimetype=mimetype)

@app.route('/<username>/stream_sum/<int:data_id>/<int:ex_id>')
def stream_sum(username=None, data_id=None,ex_id=None):
    mimetype = "text/event-stream"
    ch = username +"_summary_" + str(data_id) + "_" + str(ex_id)
    return Response(event_stream(channel=ch), mimetype=mimetype)

