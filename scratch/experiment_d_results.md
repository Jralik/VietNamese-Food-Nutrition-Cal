
# Experiment D — Rim-plane ablation results (bowl-container path)

Frozen: images/detections/masks/anchored depth/ArUco/density 0.45/integration logic (verbatim copy). Chỉ thay rim-plane knobs. Mouth plane = LSQ plane qua mask-boundary interior ring (PROXY — boundary không được giả định là physical rim).

## Mouth-plane diagnostic (per item, TRƯỚC khi tính MAPE)

| Ảnh | item | class | ring_px | min | p5 | p10 | p25 | median | max |
|---|---|---|---|---|---|---|---|---|---|
| BanhCuon1 (1).jpg | 0 | Cha | 6776 | 258 | 262 | 263 | 267 | 272 | 296 |
| BanhCuon1 (2).jpg | 0 | Cha | 2996 | 226 | 231 | 234 | 236 | 239 | 252 |
| BanhCuon1 (2).jpg | 1 | Cha | 1785 | 226 | 228 | 230 | 232 | 234 | 258 |
| BanhCuon1 (3).jpg | 0 | Cha | 2867 | 376 | 382 | 384 | 392 | 401 | 432 |
| BanhCuon1 (3).jpg | 1 | Cha | 1548 | 377 | 381 | 386 | 405 | 416 | 431 |
| BanhCuon1 (4).jpg | 0 | Cha | 6169 | 343 | 348 | 350 | 360 | 370 | 400 |
| BanhCuon1 (4).jpg | 1 | Cha | 2927 | 343 | 348 | 351 | 365 | 371 | 397 |
| BanhCuon1 (4).jpg | 2 | Cha | 2394 | 343 | 354 | 358 | 366 | 376 | 391 |
| BanhCuon1 (5).jpg | 0 | Cha | 3968 | 312 | 318 | 320 | 326 | 332 | 358 |
| BanhCuon1 (6).jpg | 0 | Cha | 2451 | 333 | 336 | 339 | 344 | 350 | 372 |
| BanhCuon1 (6).jpg | 1 | Cha | 1700 | 335 | 338 | 340 | 342 | 347 | 370 |
| BanhMiDucTri (1).jpg | 0 | Banh mi | 19462 | 265 | 280 | 285 | 293 | 310 | 382 |
| BanhMiDucTri (1).jpg | 1 | Banh mi | 6145 | 268 | 278 | 284 | 290 | 321 | 374 |
| BanhMiDucTri (2).jpg | 0 | Banh mi | 7249 | 254 | 262 | 265 | 275 | 284 | 330 |
| BanhMiDucTri (3).jpg | 0 | Banh mi | 9463 | 348 | 356 | 361 | 368 | 377 | 408 |
| BanhMi_Trung (1).jpg | 0 | Banh mi | 3536 | 468 | 477 | 478 | 483 | 488 | 556 |
| BanhMi_Trung (1).jpg | 1 | Banh mi | 2324 | 476 | 480 | 482 | 495 | 513 | 615 |
| BanhMi_Trung (1).jpg | 2 | Trung | 5442 | 478 | 488 | 494 | 505 | 512 | 554 |
| BanhMi_Trung (2).jpg | 0 | Banh mi | 3576 | 355 | 361 | 364 | 372 | 400 | 472 |
| BanhMi_Trung (2).jpg | 1 | Banh mi | 2396 | 406 | 410 | 419 | 426 | 433 | 500 |
| BanhMi_Trung (3).jpg | 0 | Banh mi | 3510 | 324 | 328 | 329 | 332 | 348 | 402 |
| BanhMi_Trung (3).jpg | 1 | Trung | 6477 | 405 | 410 | 412 | 418 | 431 | 451 |
| BanhMi_Trung (4).jpg | 0 | Banh mi | 9596 | 474 | 487 | 511 | 532 | 565 | 639 |
| BanhMi_Trung (4).jpg | 1 | Banh mi | 2620 | 510 | 516 | 520 | 560 | 589 | 638 |
| BunBoHue1 (1).jpg | 0 | Bun bo Hue | 6584 | 384 | 434 | 450 | 469 | 499 | 578 |
| BunBoHue1 (1).jpg | 1 | Rau | 3804 | 442 | 454 | 466 | 485 | 516 | 604 |
| BunBoHue1 (2).jpg | 0 | Bun bo Hue | 6682 | 500 | 511 | 516 | 534 | 557 | 626 |
| BunBoHue1 (2).jpg | 1 | Rau | 3588 | 488 | 502 | 509 | 534 | 548 | 596 |
| BunBoHue1 (3).jpg | 0 | Bun bo Hue | 8733 | 521 | 574 | 592 | 620 | 650 | 727 |
| BunBoHue1 (3).jpg | 1 | Chanh | 2198 | 650 | 662 | 670 | 694 | 730 | 792 |
| BunBoHue1 (4).jpg | 0 | Bun bo Hue | 5148 | 509 | 536 | 542 | 556 | 580 | 666 |
| BunBoHue1 (4).jpg | 1 | Chanh | 1241 | 524 | 551 | 552 | 556 | 623 | 644 |
| BunBoHue1 (4).jpg | 2 | Chanh | 466 | 584 | 586 | 587 | 590 | 596 | 614 |
| BunBoHue1 (4).jpg | 3 | Rau | 3091 | 576 | 590 | 592 | 602 | 616 | 678 |
| BunBoHue1 (5).jpg | 0 | Bun bo Hue | 8112 | 408 | 435 | 442 | 475 | 520 | 600 |
| BunBoHue1 (5).jpg | 1 | Rau | 3667 | 481 | 491 | 501 | 512 | 531 | 614 |
| Pho1 (1).jpg | 0 | Pho | 11160 | 360 | 379 | 386 | 396 | 408 | 453 |
| Pho1 (2).jpg | 0 | Pho | 17110 | 336 | 354 | 359 | 369 | 378 | 419 |
| Pho1 (3).jpg | 0 | Pho | 7886 | 454 | 465 | 472 | 484 | 501 | 562 |
| Pho1 (4).jpg | 0 | Pho | 6151 | 336 | 342 | 344 | 354 | 368 | 395 |
| Pho1 (5).jpg | 0 | Pho | 13235 | 853 | 894 | 908 | 927 | 967 | 1084 |
| SupCua1 (1).jpg | 0 | Lau | 4681 | 474 | 490 | 497 | 508 | 536 | 620 |
| SupCua1 (2).jpg | 0 | Sup cua | 6190 | 486 | 492 | 505 | 522 | 530 | 562 |
| SupCua1 (3).jpg | 0 | Canh | 5482 | 484 | 502 | 518 | 541 | 596 | 812 |
| SupCua (1).jpg | 0 | Sup cua | 6285 | 237 | 243 | 245 | 253 | 260 | 291 |
| SupCua (2).jpg | 0 | Sup cua | 5987 | 270 | 278 | 280 | 287 | 309 | 356 |
| SupCua (3).jpg | 0 | Sup cua | 4353 | 272 | 277 | 278 | 282 | 290 | 314 |
| SupCua (4).jpg | 0 | Sup cua | 3376 | 326 | 331 | 335 | 340 | 372 | 420 |
| SupCua (5).jpg | 0 | Sup cua | 5514 | 228 | 264 | 270 | 277 | 303 | 361 |
| SupCua (6).jpg | 0 | Sup cua | 7402 | 241 | 243 | 244 | 249 | 255 | 277 |
| banh-mi1.jpg | 0 | Banh mi | 10228 | 324 | 334 | 347 | 360 | 369 | 420 |
| banh-mi250g (1).jpg | 0 | Banh mi | 5833 | 393 | 406 | 410 | 424 | 430 | 506 |
| banh-mi250g (2).jpg | 0 | Banh mi | 12549 | 339 | 344 | 348 | 354 | 363 | 430 |
| banh-mi250g (3).jpg | 0 | Hamburger | 6844 | 394 | 403 | 410 | 417 | 441 | 512 |
| banh-mi250g (4).jpg | 0 | Banh mi | 3987 | 362 | 370 | 376 | 387 | 408 | 450 |
| banh-mi2.jpg | 0 | Banh mi | 17250 | 279 | 289 | 294 | 309 | 316 | 466 |
| banh-mi3.jpg | 0 | Banh mi | 15636 | 305 | 318 | 322 | 330 | 342 | 426 |
| banh-mi4.jpg | 0 | Banh mi | 12963 | 296 | 309 | 311 | 316 | 328 | 392 |
| banh-mi5.jpg | 0 | Banh mi | 29919 | 271 | 296 | 302 | 314 | 328 | 429 |
| bun-rieu1 (1).jpg | 0 | Bun rieu | 22400 | 369 | 396 | 413 | 451 | 482 | 540 |
| bun-rieu1 (2).jpg | 0 | Bun rieu | 24174 | 378 | 406 | 418 | 439 | 464 | 553 |
| bun-rieu1 (3).jpg | 0 | Bun rieu | 10041 | 290 | 320 | 330 | 350 | 378 | 468 |
| bun-rieu1 (4).jpg | 0 | Bun rieu | 11445 | 318 | 336 | 344 | 376 | 408 | 454 |
| bun-rieu (1).jpg | 0 | Bun bo Hue | 8402 | 424 | 448 | 463 | 482 | 504 | 558 |
| bun-rieu (2).jpg | 0 | Bun rieu | 6928 | 368 | 376 | 379 | 390 | 414 | 479 |
| bun-rieu (3).jpg | 0 | Bun bo Hue | 24200 | 340 | 389 | 417 | 444 | 468 | 522 |
| bun-rieu (3).jpg | 1 | Chanh | 1611 | 330 | 333 | 334 | 338 | 340 | 390 |
| com-suon1 (1).jpg | 0 | Com tam | 14710 | 389 | 412 | 422 | 442 | 492 | 581 |
| com-suon1 (2).jpg | 0 | Canh | 2576 | 524 | 529 | 532 | 536 | 542 | 590 |
| com-suon1 (2).jpg | 1 | Com tam | 8787 | 504 | 523 | 532 | 547 | 569 | 614 |
| com-suon1 (3).jpg | 0 | Com tam | 5869 | 652 | 664 | 670 | 674 | 683 | 788 |
| com-suon1 (3).jpg | 1 | Com tam | 3269 | 496 | 508 | 512 | 516 | 524 | 560 |
| com-suon1 (4).jpg | 0 | Canh | 2133 | 387 | 408 | 418 | 430 | 447 | 502 |
| com-suon1 (4).jpg | 1 | Com tam | 9181 | 356 | 366 | 370 | 381 | 399 | 504 |
| com-suon1 (5).jpg | 0 | Canh | 1834 | 308 | 310 | 312 | 329 | 335 | 351 |
| com-suon1 (5).jpg | 1 | Com tam | 17740 | 284 | 301 | 311 | 328 | 350 | 460 |

## Bảng 1 — Mass MAPE per class per variant (target-matched, %)

| Variant | Banh mi | Bun bo Hue | Bun rieu | Canh | Com tam | Pho | Rau | Trung | Sup cua |
|---|---|---|---|---|---|---|---|---|---|
| D0 | 87.2% (15) | 11.0% (5) | 11.0% (5) | 200.9% (1) | 10.1% (5) | 21.9% (5) | 33.3% (4) | 66.7% (2) | 61.3% (7) |
| D1 | 87.2% (15) | 10.7% (5) | 8.6% (5) | 232.1% (1) | 10.1% (5) | 22.3% (5) | 33.3% (4) | 66.7% (2) | 67.3% (7) |
| D2 | 87.2% (15) | 11.0% (5) | 11.9% (5) | 200.9% (1) | 10.1% (5) | 21.9% (5) | 33.3% (4) | 66.7% (2) | 61.3% (7) |
| D3 | 87.2% (15) | 11.0% (5) | 11.0% (5) | 200.9% (1) | 10.1% (5) | 21.9% (5) | 33.3% (4) | 66.7% (2) | 61.3% (7) |
| D4 | 87.2% (15) | 10.7% (5) | 8.0% (5) | 232.1% (1) | 10.1% (5) | 22.3% (5) | 33.3% (4) | 66.7% (2) | 67.3% (7) |

## Bảng 2 — Raw vs clipped volume + clip-hit (SupCua nhóm)

| Variant | Raw volume median (cm³) | Clipped volume median (cm³) | Clip-hit rate |
|---|---|---|---|
| D0 | 451 | — | 1/76 |
| D1 | 451 | — | 3/76 |
| D2 | 451 | — | 1/76 |
| D3 | 451 | — | 1/76 |
| D4 | 451 | — | 2/76 |

## Per-image SupCua volume per variant (cm³) — GT cột GT

| Ảnh | GT | D0 | D1 | D2 | D3 | D4 |
|---|---|---|---|---|---|---|
| SupCua1 (1).jpg | 501 | 1130 | 1130 | 1130 | 1130 | 1130 |
| SupCua1 (2).jpg | 501 | 619 | 619 | 619 | 619 | 619 |
| SupCua1 (3).jpg | 501 | 619 | 619 | 619 | 619 | 619 |
| SupCua (1).jpg | 471 | 451 | 451 | 451 | 451 | 451 |
| SupCua (2).jpg | 471 | 451 | 451 | 451 | 451 | 451 |
| SupCua (3).jpg | 471 | 1188 | 1200 | 1188 | 1188 | 1200 |
| SupCua (4).jpg | 471 | 1200 | 1200 | 1200 | 1200 | 1200 |
| SupCua (5).jpg | 471 | 824 | 1006 | 824 | 824 | 1006 |
| SupCua (6).jpg | 471 | 451 | 451 | 451 | 451 | 451 |

