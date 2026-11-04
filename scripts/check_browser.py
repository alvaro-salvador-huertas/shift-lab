import argparse,asyncio,json,re,shutil
from datetime import datetime,timezone
from pathlib import Path
from playwright.async_api import async_playwright,expect

async def main():
 parser=argparse.ArgumentParser(description='QA local de ShiftLab con Chrome existente; conserva capturas y verifica JSON.');parser.add_argument('--source',default='artifacts/validation/results.json');parser.add_argument('--tag');parser.add_argument('--url',default='http://127.0.0.1:8503');args=parser.parse_args()
 root=Path(__file__).resolve().parents[1]
 if args.tag is None: args.tag='qa-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')
 if not re.fullmatch(r'[A-Za-z0-9_-]+',args.tag): parser.error('Usa letras, números, guiones o guion bajo en --tag.')
 if (root/'artifacts'/f'browser-{args.tag}.json').exists(): parser.error('El informe ya existe; usa otra etiqueta para conservarlo.')
 chrome=shutil.which('google-chrome') or shutil.which('chromium') or shutil.which('chromium-browser')
 if chrome is None: parser.error('Se necesita Chrome/Chromium instalado. El script no descarga navegadores.')
 shots=root/'docs/screenshots';shots.mkdir(parents=True,exist_ok=True)
 if list(shots.glob(f'{args.tag}-*.png')): parser.error('Ya hay capturas con esta etiqueta; usa otra.')
 payload=json.loads((root/args.source).read_text());errors=[];external=[]
 async with async_playwright() as p:
  browser=await p.chromium.launch(executable_path=chrome,headless=True,args=['--no-sandbox','--disable-gpu'])
  ctx=await browser.new_context(viewport={'width':1440,'height':1000},accept_downloads=True)
  async def route(r):
   if r.request.url.startswith((args.url.rstrip('/')+'/',args.url.replace('http:','ws:').rstrip('/')+'/','data:','blob:')): await r.continue_()
   else: external.append(r.request.url);await r.abort()
  await ctx.route('**/*',route);page=await ctx.new_page();page.on('pageerror',lambda e:errors.append(str(e)))
  await page.goto(args.url);await expect(page.get_by_role('button',name='Cargar resultados',exact=True)).to_be_visible(timeout=30000)
  await page.get_by_role('textbox',name='Archivo de resultados').fill(args.source);await page.get_by_role('textbox',name='Archivo de resultados').press('Enter')
  await page.get_by_role('button',name='Cargar resultados',exact=True).click()
  await expect(page.get_by_role('heading',name='La misma evaluación para todos los candidatos')).to_be_visible(timeout=30000)
  scenario=page.get_by_role('combobox',name='Condición de evaluación');await scenario.click();await page.get_by_role('option',name='Ruido · σ 0,05',exact=True).click()
  await expect(page.get_by_text('Candidato seleccionado:',exact=False)).to_contain_text('σ 0,05',timeout=30000)
  await expect(page.get_by_test_id('stStatusWidget')).to_be_hidden(timeout=30000);await page.wait_for_timeout(350);await page.screenshot(path=str(shots/f'{args.tag}-resumen-desktop.png'))
  async with page.expect_download(timeout=30000) as d:
   await page.get_by_role('button',name='Descargar resultados · JSON',exact=True).click()
  download=await d.value;destination=root/'artifacts'/f'browser-{args.tag}-download.json';await download.save_as(destination)
  assert json.loads(destination.read_text())==payload
  await page.get_by_text('Explorador',exact=True).click();await expect(page.get_by_role('heading',name='Mira qué recibió el clasificador')).to_be_visible(timeout=30000)
  await expect(page.locator('[data-testid="stImage"]')).to_have_count(2,timeout=30000)
  await page.get_by_role('heading',name='Mira qué recibió el clasificador').scroll_into_view_if_needed()
  await expect(page.get_by_test_id('stStatusWidget')).to_be_hidden(timeout=30000);await page.wait_for_timeout(350);await page.screenshot(path=str(shots/f'{args.tag}-explorador-desktop.png'))
  await page.get_by_role('combobox',name='Imagen · índice original').click();await page.keyboard.press('ArrowDown');await page.keyboard.press('Enter')
  await expect(page.locator('[data-testid="stImage"]')).to_have_count(2,timeout=30000)
  await page.get_by_text('Análisis de errores',exact=True).click();await expect(page.get_by_role('heading',name='Qué clases se confunden')).to_be_visible(timeout=30000)
  await expect(page.get_by_test_id('stStatusWidget')).to_be_hidden(timeout=30000);await page.wait_for_timeout(350);await page.screenshot(path=str(shots/f'{args.tag}-errores-desktop.png'))
  await page.set_viewport_size({'width':390,'height':844})
  collapse=page.get_by_test_id('stSidebarCollapseButton')
  await page.wait_for_timeout(600)
  await page.evaluate("const s=document.querySelector('[data-testid=stSidebar]'); if(s && s.getBoundingClientRect().right>50){document.querySelector('[data-testid=stSidebarCollapseButton] button')?.click();}")
  await page.wait_for_timeout(500)
  await expect(page.get_by_test_id('stStatusWidget')).to_be_hidden(timeout=30000);await page.wait_for_timeout(350);await page.screenshot(path=str(shots/f'{args.tag}-errores-mobile.png'))
  await page.get_by_text('Explorador',exact=True).click();await expect(page.get_by_role('heading',name='Mira qué recibió el clasificador')).to_be_visible(timeout=30000)
  await expect(page.get_by_test_id('stStatusWidget')).to_be_hidden(timeout=30000);await page.wait_for_timeout(350);await page.screenshot(path=str(shots/f'{args.tag}-explorador-mobile.png'))
  await page.get_by_text('Resumen y comparación',exact=True).click();await expect(page.get_by_role('heading',name='La misma evaluación para todos los candidatos')).to_be_visible(timeout=30000)
  await expect(page.get_by_test_id('stStatusWidget')).to_be_hidden(timeout=30000);await page.wait_for_timeout(350);await page.screenshot(path=str(shots/f'{args.tag}-resumen-mobile.png'))
  await page.get_by_test_id('stPlotlyChart').first.screenshot(path=str(shots/f'{args.tag}-comparacion-mobile-detail.png'))
  await page.get_by_test_id('stPlotlyChart').nth(1).screenshot(path=str(shots/f'{args.tag}-curva-mobile-detail.png'))
  overflow=await page.evaluate('({width:innerWidth,scroll:document.documentElement.scrollWidth})')
  assert overflow['scroll']<=overflow['width']+1,overflow
  report={'created_at':datetime.now(timezone.utc).isoformat(),'stage':payload['stage'],'source':args.source,'browser':browser.version,'desktop':[1440,1000],'narrow':[390,844],'views':['Resumen y comparación','Explorador','Análisis de errores'],'download_matches_source':True,'page_errors':errors,'external_requests_blocked':external,'overflow':overflow,'screenshots':sorted(x.name for x in shots.glob(f'{args.tag}-*.png'))}
  (root/'artifacts').mkdir(exist_ok=True)
  with (root/'artifacts'/f'browser-{args.tag}.json').open('x',encoding='utf-8') as handle: handle.write(json.dumps(report,ensure_ascii=False,indent=2))
  print(json.dumps(report,ensure_ascii=False,indent=2));await browser.close()
asyncio.run(main())
