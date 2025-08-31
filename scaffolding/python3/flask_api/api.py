from flask import Flask, jsonify, request
from flask_basicauth import BasicAuth

app = Flask(__name__)
basic_auth = BasicAuth(app)

# auth config
app.config['BASIC_AUTH_USERNAME'] = 'dummy_user'
app.config['BASIC_AUTH_PASSWORD'] = 'dummy_password'

key_values = {'k1': 'v1',
              'k2': 'v2'}


# get all key values
@app.route('/api/v1/key_values', methods=['GET'])
def get_key_values():
    return jsonify(key_values), 200


# get a key's value
@app.route('/api/v1/key_values/<key>', methods=['GET'])
def get_value(key):
    if key not in key_values:
        return jsonify({'error': "Key not found"}), 404
    return jsonify({key: key_values[key]})


# create new key_value pair
@app.route('/api/v1/key_values', methods=['POST'])
def create_key_value():
    if (not request.json or 'key' not in request.json
       or 'value' not in request.json):
        return jsonify({'error': 'Key and value are required'}), 400
    key_values[request.json['key']] = request.json['value']
    return jsonify({request.json['key']: request.json['value']}), 201


# update an existing key
@app.route('/api/v1/key_values/<key>', methods=['PUT'])
def update_key_value(key):
    if key not in key_values:
        return jsonify({'error': 'Key not found'}), 404
    if not request.json or 'value' not in request.json:
        return jsonify({'error': 'No value provided'}), 400
    key_values[key] = request.json['value']
    return jsonify({key: request.json['value']})


# delete a key_value pair
@app.route('/api/v1/key_values/<key>', methods=['DELETE'])
def delete_key_value(key):
    if key not in key_values:
        return jsonify({'error': 'Key not found'}), 404
    del key_values[key]
    return jsonify({'result': 'success'}), 200


# uses HTTP Basic Authentication
# auth sent in header
@app.route('/api/v1/auth')
@basic_auth.required
def protected():
    return jsonify(message='You are authenticated'), 200


if __name__ == '__main__':
    app.run(debug=True)
