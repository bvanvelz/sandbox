## Example Requests

CREATE
```
curl -X POST --json "{\"key\": \"k_new\", \"value\": \"value_new\"}" http://127.0.0.1:5000/api/v1/key_values
```

READ
```
curl -X GET http://127.0.0.1:5000/api/v1/key_values

curl -X GET http://127.0.0.1:5000/api/v1/key_values/k1
```

UPDATE
```
curl -X PUT --json "{\"value\": \"value_new\"}" http://127.0.0.1:5000/api/v1/key_values/k2
```

DELETE
```
curl -X DELETE http://127.0.0.1:5000/api/v1/key_values/k1
```

AUTH
```
curl -u dummy_user:dummy_password http://127.0.0.1:5000/api/v1/auth
```
