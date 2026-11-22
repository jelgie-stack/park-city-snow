# Park City Snow Outlook

A one-page, phone-friendly snow outlook for Park City, Utah: **https://jelgie-stack.github.io/park-city-snow/**

- **Next 10 days:** chance of snow each day and storm timing at the Park City Mountain summit (9,715 ft) and in town (7,000 ft), from 154 ensemble forecasts (ECMWF, ECMWF AI, GFS ensemble, Canada GEM) via [Open-Meteo](https://open-meteo.com).
- **This season:** Thaynes Canyon SNOTEL snowpack vs the 1991–2020 normal and the 2022–23 record year, with a 10-day projection ([NRCS](https://wcc.sc.egov.usda.gov/nwcc/site?sitenum=814)).
- **Longer range:** GFS ensemble weeks 1–4 and NOAA [Climate Prediction Center](https://www.cpc.ncep.noaa.gov/) outlooks.

## How it works

- `index.html`: the page (vanilla JS, no dependencies). It reads `data.json`.
- `build.py`: pulls the data and writes `data.json`.
- `.github/workflows/refresh.yml`: runs `build.py` twice a day and commits the new `data.json`. GitHub Pages serves the result.

Run it locally:

```bash
pip install requests numpy pandas pyshp shapely
python build.py .
python -m http.server   # then open http://localhost:8000
```

Forecasts 8–10 days out are uncertain; see the "How to read this" section on the page. Not an avalanche or travel forecast.
