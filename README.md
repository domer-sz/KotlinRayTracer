# Kotlin Ray Tracer

Path tracer z książek *Ray Tracing in One Weekend* / *The Next Week* / *The Rest of Your Life*,
napisany w Kotlinie, liczony na GPU przez CUDA, z odszumianiem Intel Open Image Denoise
i interaktywnym oknem podglądu.

## Wymagania

- **Windows x64** (biblioteki odszumiacza i okno trzymające zegary GPU są przygotowane pod Windows;
  sam render działa też na Linuksie).
- **Karta NVIDIA z CUDA** – testowane na GTX 1070 (Pascal, sm_61) ze sterownikiem 577. Kompilator
  kernela (NVRTC) jest w `native/cuda`, więc CUDA Toolkit nie jest potrzebny. Bez karty NVIDIA renderer
  nie działa (nie ma już trybu CPU ani OpenCL).
- **JDK 17 lub nowszy** (testowane na Temurin 21) i **Maven**.

Nic więcej nie trzeba instalować: odszumiacz OIDN jest w repozytorium (`native/oidn/windows-x64`).

## Uruchomienie interfejsu graficznego

Z katalogu repozytorium:

```
mvn compile
mvn exec:java -Drt.gui -Drt.scene=cornell-final
```

Otworzy się okno ze sceną w **szybkim podglądzie** – trybie do ustawiania sceny: jeden promień na
piksel, kolory powierzchni bez odbić i cieni, szkło półprzezroczyste (widać, co stoi za kulą).
Obraz jest od razu ostry w pełnej rozdzielczości, a przesuwanie, obracanie i skalowanie modeli –
także takich z setkami tysięcy trójkątów – odświeża się w ułamku sekundy.

Klawisz **T** (albo *Tryb* w panelu) przełącza na **ray tracing**: pełne oświetlenie, szkło, cienie,
obraz wyostrza się w miarę zbierania próbek i jest odszumiany. `-Drt.gui.raytrace` otwiera okno
od razu w tym trybie.

Kilka przydatnych wariantów:

```
# większe okno
mvn exec:java -Drt.gui -Drt.scene=cornell-final -Drt.width=900

# od razu z modelem na scenie (można podać kilka, rozdzielonych średnikiem)
mvn exec:java -Drt.gui -Drt.scene=cornell-final "-Drt.gui.models=../stls/witcher/witcher_assembled.stl"

# z włączonym światłem przednim i kolumną w szachownicę
mvn exec:java -Drt.gui -Drt.scene=cornell-final -Drt.fill=0.4 -Drt.block=checker
```

Duże modele (setki tysięcy – miliony trójkątów) potrzebują więcej pamięci dla Javy.
W PowerShell przed uruchomieniem:

```
$env:MAVEN_OPTS = "-Xmx12g"
```

(w bashu: `export MAVEN_OPTS=-Xmx12g`).

Inne sceny: `bouncing-spheres`, `earth`, `perlin-spheres`, `quads`, `simple-light`, `cornell-box`,
`cornell-blocks`, `cornell-smoke`, `final-week`, `cornell-final`, `showcase`, `cornell-model`, `model`.
Najwięcej funkcji okna (światła do przełączania, obiekty sceny do usunięcia) mają `cornell-final`
i `cornell-model`.

## Sterowanie

Pełna lista jest też w oknie pod **F1** (albo **H**, albo przycisk *Pomoc* w panelu).

| Akcja | Skrót |
|---|---|
| Obrót kamery wokół punktu, na który patrzy | lewy przycisk myszy + przeciąganie |
| Przesuwanie kamery | Shift + lewy przycisk + przeciąganie, albo środkowy przycisk |
| Przybliżanie / oddalanie | kółko myszy albo prawy przycisk + przeciąganie |
| Szybki podgląd / ray tracing | `T` |
| Kąt widzenia | `+` / `-` |
| Reset widoku | `R` |
| Przestaw zaznaczony model – żółty zarys pokazuje, gdzie stanie (podłoga, kolumna, kula, inny model; przy ścianie u jej stóp) | Ctrl + przeciąganie albo Ctrl + klik |
| Przesuń zaznaczony model w lewo / prawo / dalej / bliżej (względem kamery) | strzałki (Shift = krok 5× większy) |
| Podnieś / opuść zaznaczony model | PageUp / PageDown |
| Obróć zaznaczony model o 15° | `Q` / `E` albo `[` / `]` |
| Usuń zaznaczony obiekt | `Delete` |
| Odszumianie włącz / wyłącz | `D` |
| Zapisz obraz pod wybraną nazwą (PPM + PNG, z odszumioną kopią) | `S` |
| Zapisz scenę / wczytaj scenę (`.rtscene`) | `Ctrl+S` / `Ctrl+O` |
| Pomoc | `F1` / `H` |
| Zamknij | `Esc` |

W trybie ray tracing podczas ruchu kamery podgląd liczy się w 1/4 rozdzielczości przy 1 próbce na
piksel, żeby był płynny. Po zatrzymaniu (150 ms) obraz przechodzi na pełną rozdzielczość, zbiera
próbki do celu ustawionego w panelu i jest odszumiany przy 16, 64, 256… próbkach oraz na końcu.

## Panel boczny

- **Scena** – *Zapisz…* / *Wczytaj…* (Ctrl+S / Ctrl+O): cała scena w pliku tekstowym `.rtscene`.
- **Tryb** – szybki podgląd albo ray tracing (to samo co `T`).
- **Obraz** – gotowe rozdzielczości albo własna szerokość i wysokość (*Zastosuj rozdzielczość*),
  **Cel spp** (liczba próbek, przy której render się kończy) z paskiem postępu, **Odbicia (maks.)**
  – ile razy promień może się odbić lub załamać (domyślnie wartość sceny, np. 50; mniej = szybciej,
  ale ciemniej w szkle i zakamarkach), odszumianie, zapis.
- **Światła** – lampa sufitowa i światło przednie do włączania i wyłączania.
- **Obiekty na scenie** – lista obiektów: kolumna i szklana kula sceny (można je przesuwać i obracać
  tak jak dodane obiekty – polami X / Y / Z i Obrót, strzałkami, Q / E, Ctrl + przeciąganiem – usunąć i przywrócić
  przyciskiem *Przywróć obiekty sceny*) oraz dodane obiekty. *Dodaj model…* wczytuje STL, OBJ, PLY,
  OFF, glTF/GLB albo 3MF; *Dodaj kulę* i *Dodaj graniastosłup* wstawiają kolejną kulę (szklaną)
  albo graniastosłup w proporcjach kolumny (biały, z własną szerokością i głębokością). Zaznaczony
  obiekt można ustawić polami X / Y / Z, obrotem, wysokością i materiałem, przeciągnąć z Ctrl
  albo przesuwać strzałkami. Materiał to gotowy zestaw (szary, biały, czerwony
  i niebieski mat, szkło, diament, lustro, złoto, miedź, stal szczotkowana, świecący biały) albo własny:
  rodzaj (matowy, metal, szkło, świecący), kolor z okna wyboru i parametr zależny od rodzaju
  (chropowatość metalu, współczynnik załamania szkła, moc światła).
  Na końcu listy są ściany pokoju (lewa, prawa, tylna, podłoga, sufit): po zaznaczeniu można
  zmienić ich materiał tym samym edytorem, np. zrobić lustrzaną ścianę; *Materiał jak w scenie*
  przywraca oryginał.
  *Oś Z w górę* włącza się sama dla plików do druku 3D (STL, 3MF).
  Model OBJ z plikiem materiałów (`.mtl`) wczytuje się w swoich kolorach – *Gotowy: Kolory z pliku*;
  wybranie innego materiału zastępuje je, a *Kolory z pliku* do nich wraca. Czytane są `Kd` (kolor),
  `Pm`/`Pr` (metal, chropowatość), `Ke` (świecenie) i `d`/`Ni` (szkło). Przykład: `models/orc-painted.obj`.

## Zapis obrazu i sceny

**Obraz** (`S` albo *Zapisz obraz*): okno zapisu podpowiada `viewer-<scena>-<data>` w katalogu
roboczym (albo tam, gdzie wskazuje `-Drt.out`); można wpisać dowolną nazwę i wybrać folder.
Powstają `<nazwa>.ppm` i `<nazwa>.png`, a w trybie ray tracing z odszumianiem także
`<nazwa>-denoised.ppm/.png`. Końcówka `.png`/`.ppm` w nazwie jest pomijana, a istniejące pliki
nadpisują się dopiero po potwierdzeniu.

**Scena** (`Ctrl+S` / `Ctrl+O` albo przyciski w sekcji *Scena*): plik `.rtscene` zapamiętuje
scenę bazową (z przełącznikami takimi jak `-Drt.block=checker`), kamerę, rozdzielczość, cel spp,
tryb, odszumianie, włączone światła, usunięte obiekty sceny oraz każdy dodany obiekt: kształt,
położenie, obrót, wymiary, oś Z i materiał. Modele zapisuje jako ścieżki do plików, względne wobec
pliku sceny, więc scena zapisana w repozytorium otworzy się z innej kopii. To zwykły tekst – można
go poprawiać ręcznie.

Scenę można też otworzyć od razu przy starcie (scena bazowa i przełączniki są brane z pliku):

```
mvn exec:java "-Drt.gui.load=moja-scena.rtscene"
```

## Render do pliku (bez okna)

Bez `-Drt.gui` program renderuje scenę do pliku i kończy działanie:

```
mvn exec:java -Drt.scene=cornell-final -Drt.width=2000 -Drt.samples=2000 -Drt.out=render.ppm
```

Powstają `render.ppm`, `render.png` i odszumione `render-denoised.ppm/.png`
(`-Drt.denoise=off` wyłącza odszumianie). Pozostałe atrybuty – rozdzielczość, próbki, modele,
materiały, światła – opisuje `AGENTS.md`.

## Rozwiązywanie problemów

- **Czarne okno albo „GPU rendering unavailable”** – brak karty NVIDIA z CUDA albo sterownika; zaktualizuj
  sterownik karty. „NVRTC not found” – uruchomiono spoza katalogu repozytorium; wskaż `native/cuda/windows-x64`
  przez `-Drt.nvrtc=<ścieżka>`.
- **„bez odszumiania (brak OIDN)” w pasku stanu** – brakuje `native/oidn/windows-x64`
  (np. uruchomiono spoza katalogu repozytorium); wskaż go przez `-Drt.oidn=<ścieżka>`.
- **OutOfMemoryError przy dużym modelu** – zwiększ `MAVEN_OPTS` (`-Xmx12g` lub więcej).
