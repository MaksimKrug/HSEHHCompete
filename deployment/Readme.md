## Docker build
```
docker build --rm -t hh_compete .
```

## Docker run
```
docker run -v $(pwd)/data/:/app/data hh_compete
```