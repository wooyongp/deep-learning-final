library(tidyverse)
library(haven)
library(fixest)
library(data.table)

# COVID-19 Susceptibility and Government Sentiment 
# Were more susceptible but non
setwd("final-project")

d23 <- read_dta('data/klips/klips23p.dta')  %>% as.data.table()
d23hh <- read_dta('data/klips/klips23h.dta') %>% as.data.table()

d23 <- d23hh[d23, on=.(hhid23)]

d23a <- read_dta('data/klips/klips23a.dta') %>% as.data.table()

d23 <- d23a[d23, on=.(pid)]
rm(d23hh, d23a)

d23[, had_covid := fifelse(!is.na( a232401 ), 1, 0), by=.(hhid23)]
d23[, n_distinct(hhid23), by=.(had_covid)]

d23[, n_distinct(hhid23), by=.(a232402)]
