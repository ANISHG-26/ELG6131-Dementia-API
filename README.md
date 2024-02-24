# API for Dementia Prediction

This Flask API has been created to serve an endpoint to requests that would predict dementia based on brain MRI images. ML model used is a tensorflow CNN.

# Heroku Deployment Guide

## Docker Commands

### Build container image for Heroku

`docker build -t registry.heroku.com/dementia-prediction/web .`

### Local container run

`docker run -p 5000:5000 --name Dementia-Prediction registry.heroku.com/dementia-prediction/web`

## Heroku Commands

### Heroku Login

`heroku login`

### Heroku Container Registry Login

`heroku container:login`

### Push image to Heroku Container Registry

`docker push registry.heroku.com/dementia-prediction/web`

### Push image for Heroku App

`heroku container:push web -a dementia-prediction`

### Release Application

`heroku container:release web -a dementia-prediction`

### Scale to save money

`heroku ps:scale web=0 -a dementia-prediction`

### Scale to lose money

`heroku ps:scale web=1 -a dementia-prediction`
